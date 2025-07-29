# -*- coding: utf-8 -*-
"""
GRPO fine‑tuning of Qwen2.5‑VL‑3B on CheXpert multi‑label classification
-----------------------------------------------------------------------
This script continues from the user‑supplied cells and adds:
  • dataset preprocessing (ground‑truth label extraction + prompt building)
  • two reward functions (format + classification)
  • a composite reward_fn compatible with Unsloth GRPOTrainer
  • GRPO training loop and LoRA export

Run in environment with: pip install unsloth "scikit-learn>=1.4" datasets tensorboard
"""
import os 
import torch
num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs")


# vLLM/TRL colocation still wants DDP-style env vars even on 1 node
#os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
#os.environ.setdefault("MASTER_PORT", "29500")
#os.environ.setdefault("WORLD_SIZE", "1") # str(num_gpus)
#os.environ.setdefault("RANK", "0")
#os.environ.setdefault("LOCAL_RANK", "0")
#os.environ.setdefault("VLLM_WORKER_MULTIPROC", "0")  # keep vLLM single-proc in colocate

#os.environ.setdefault("TRANSFORMERS_SKIP_TP_INIT", "1")   # stop HF TP auto-init
#os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")      # silence/remove bnb
#os.environ.setdefault("VLLM_WORKER_MULTIPROC", "0")       # keep vLLM in-proc

os.environ["HF_TOKEN"] = ""
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir

os.environ["VLLM_CACHE_ROOT"] = os.path.join(cache_dir, "vllm")
os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = "spawn"
os.environ['TOKENIZERS_PARALLELISM'] = "True"

from huggingface_hub import login
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    print("✅ Authenticated with HuggingFace")
    
import re, torch, os
import argparse
from collections import defaultdict
from peft import LoraConfig, get_peft_model
from datetime import datetime
# Set Hugging Face cache to current working directory


from datasets import load_dataset, config
from transformers.utils import TRANSFORMERS_CACHE
print(f"Transformers cache: {TRANSFORMERS_CACHE}")
print(f"Cache directory: {os.environ.get('HF_HOME')}")
from sklearn.metrics import f1_score

#from unsloth import FastModel, FastVisionModel
from trl import GRPOConfig, GRPOTrainer
from vision_GRPO_trainer import Qwen2VLGRPOTrainer
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info



# 0.  GLOBAL HYPERPARAMETERS & CONFIGURATIONS  ─────────────────────────
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME =  "Qwen/Qwen2.5-VL-3B-Instruct"  # "google/medgemma-4b-it"  # "google/gemma-3-4b-it"
DEFAULT_N_SAMPLES_TRAIN = 2000  # Number of samples for quick testing
DEFAULT_MAX_STEPS = 1000  # Maximum training steps
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_GENERATIONS = 3
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 4  # 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_OUTPUT_DIR = "outputs_v3_test_vllm"
DEFAULT_TENSORBOARD_DIR = "tensorboard"
DEFAULT_SAVE_STEPS = 50

# CheXpert classification labels
LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

# Global regex patterns (defined once for efficiency)
label_regex = re.compile(
    r"|".join([re.escape(c) for c in LABEL_COLS]), re.IGNORECASE
)
answer_regex = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
NO_NESTED_TAGS_CONTENT_PATTERN = r"(?:(?!</?(?:thinking|answer)>).)*?"
structure_regex = re.compile(
    rf"^\s*<thinking>({NO_NESTED_TAGS_CONTENT_PATTERN})</thinking>\s*<answer>({NO_NESTED_TAGS_CONTENT_PATTERN})</answer>\s*$",
    re.IGNORECASE | re.DOTALL
)

new_tokens = ["<thinking>", "</thinking>", "<answer>", "</answer>"]

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a VLM on CheXpert with GRPO.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the pre-trained model to use.")
    parser.add_argument("--n_samples_train", type=int, default=DEFAULT_N_SAMPLES_TRAIN, help="Number of training samples to use (for quick testing). Set to -1 to use full training set.")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum training steps.")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--num_generations", type=int, default=DEFAULT_NUM_GENERATIONS, help="Number of generations per prompt in GRPO.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE, help="Training batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps.")
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R, help="LoRA r dimension.")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save all outputs (checkpoints, logs, final model).")
    parser.add_argument("--tensorboard_dir", type=str, default=DEFAULT_TENSORBOARD_DIR, help="Tensorboard logs directory (relative to output_dir).")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS, help="Save checkpoint every N steps.")
    
    return parser.parse_args()

# ---------------------------------------------------------------------
# Helper functions for prompt preparation and label extraction
# ---------------------------------------------------------------------

def prepare_prompt(sample, processor_ref, instruction_text):
    """Build VLM chat prompt compatible with unsloth/Qwen template"""
    conversation = [  
            {
            "role": "system",
            "content": [{"type": "text", "text": "You are a radiologist analyzing a chest X-ray. Provide your complete diagnostic reasoning step by step between <thinking> </thinking> tags, then your final diagnosis in <answer> </answer> tags"}]
            },
            {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": instruction_text},
            ],
            },
    ]
    sample["prompt"] = processor_ref.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    return sample

def extract_gt(sample):
    """Extract ground-truth labels (correctly identify 'present' labels with value 3)"""
    # Uncertain (1) and Absent (2) are treated as not present for binary classification
    sample["gt_labels"] = [c for c in LABEL_COLS if sample[c] == 3]
    return sample

# ---------------------------------------------------------------------
# 3.  REWARD FUNCTIONS  ────────────────────────────────────────────────
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Reward utility functions
# ---------------------------------------------------------------------
def parse_pred(text: str):
    """Extract text within <answer> tags and return unique list of category names found"""
    match = answer_regex.search(text)
    if not match:
        return []
    content = match.group(1).strip()
    return list({m.group(0).title() for m in label_regex.finditer(content)})

# ---------------------------------------------------------------------
# Separate reward functions for GRPOTrainer
# ---------------------------------------------------------------------
def reward_format(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    +1.0 when output matches <thinking>...</thinking><answer>...</answer> with non-empty answer,
    -0.5 when structure correct but empty answer,
    -1.0 otherwise.
    """
    print(f"DEBUG reward_format:")
    print(f"  len(prompts): {len(prompts)}")
    print(f"  len(completions): {len(completions)}")

    rewards = []
    for idx, text in enumerate(completions):
        match = answer_regex.search(text)
        if structure_regex.search(text) and match and match.group(1).strip():
            r = 1.0
        elif structure_regex.search(text):
            r = -0.5
        else:
            r = -1.0
        print(f"[Format] idx={idx} reward={r}")
        print(f"[Format] idx={idx} Generated Text:\n{text}")
        rewards.append(r)
    print(f"  len(rewards) returned: {len(rewards)}")
    return rewards


def reward_classification(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Scaled sample-level F1 reward: 2*F1(samples)-1, where F1 is computed with average='samples'.
    """


    #TODO: check if gt_labels is a list of lists, where each sublist corresponds to the ground truth labels for each prompt
    gt_list = kwargs.get("gt_labels", [])
    print(f"DEBUG reward_classification:")
    print(f"  len(prompts): {len(prompts)}")
    print(f"  len(completions): {len(completions)}")
    print(f"  len(gt_list): {len(gt_list)}")
    # gt_list contiene le etichette per ogni prompt unico
    # ma completions contiene num_generations completions per ogni prompt
    # quindi dobbiamo replicare gt_labels per ogni generazione
    
    num_generations = len(completions) // len(gt_list)
    print(f"  num_generations calculated: {num_generations}")
    expanded_gt_list = []
    for gt_labels in gt_list:
        for _ in range(num_generations):
            expanded_gt_list.append(gt_labels)

    print(f"  len(expanded_gt_list): {len(expanded_gt_list)}")
    rewards = []

    for idx, (text, gt_labels) in enumerate(zip(completions, expanded_gt_list)):
        pred_labels = parse_pred(text)
        y_true = [1 if lbl in gt_labels else 0 for lbl in LABEL_COLS]
        y_pred = [1 if lbl in pred_labels else 0 for lbl in LABEL_COLS]
        f1_sample = f1_score([y_true], [y_pred], average="samples", zero_division=0)
        r = 2 * f1_sample - 1
        print(f"[Class] idx={idx} GT={gt_labels} PRED={pred_labels} f1={f1_sample:.3f} reward={r:.3f}")
        rewards.append(r)
    print(f"  len(rewards) returned: {len(rewards)}")
    return rewards


def reward_exact_match(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Bonus reward: +1.0 if predicted set of labels exactly equals ground truth set, 0.0 otherwise.
    # This adds a discrete jump that encourages perfect matches but may destabilize training.
    """
    gt_list = kwargs.get("gt_labels", [])
    rewards = []
    for idx, (text, gt_labels) in enumerate(zip(completions, gt_list)):
        pred_set = set(parse_pred(text))
        gt_set = set(gt_labels)
        r = 1.0 if pred_set == gt_set else 0.0
        print(f"[Exact] idx={idx} match={pred_set == gt_set} reward={r}")
        rewards.append(r)
    return rewards

# ---------------------------------------------------------------------
# MAIN SCRIPT LOGIC  ──────────────────────────────────────────────────
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    # set these variables to use multi-GPU vllm inference during training
    #if num_gpus > 1:
    #    # Configurazione per multi-GPU
    #    os.environ["RANK"] = "0"
    #    os.environ["LOCAL_RANK"] = "0"
    #    os.environ["WORLD_SIZE"] = "1"
    #    os.environ["MASTER_ADDR"] = "localhost"
    #    os.environ["MASTER_PORT"] = "29500"
    #    device_map = "auto"
    # Create experiment-specific directory structure
    # Extract model name without path separators for folder name
    model_name_clean = args.model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"{model_name_clean}_{timestamp}"
    
    # Create full experiment path
    full_experiment_path = os.path.join(args.output_dir, experiment_dir)
    checkpoint_dir = os.path.join(full_experiment_path, "checkpoints")
    tensorboard_path = os.path.join(full_experiment_path, "tensorboard")
    final_model_path = os.path.join(full_experiment_path, "final_lora_adapter")
    
    # Create all necessary directories
    os.makedirs(full_experiment_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    print(f"Experiment directory: {full_experiment_path}")
    print(f"Checkpoints directory: {checkpoint_dir}")
    print(f"Tensorboard logs: {tensorboard_path}")
    print(f"Final model will be saved to: {final_model_path}")

    # ---------------------------------------------------------------------
    # 1.  LOAD MODEL ─────────────────────────────────────────────────────
    # ---------------------------------------------------------------------
    print(f"Loading model: {args.model_name}")
    
    # Load model with correct class and recommended settings
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Enable if needed for better performance
    )

    # Load processor and tokenizer separately as recommended
    # Configure processor with specific parameters for training compatibility
    min_pixels = 128 * 28 * 28  # Reduced for training stability
    max_pixels = 720 * 28 * 28  # Reduced for training stability
    processor = AutoProcessor.from_pretrained(
        args.model_name, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set pad_token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to: {tokenizer.pad_token}")

    # Add new special tokens for thinking and answer tags
    tokenizer.add_tokens(new_tokens, special_tokens=False)
    model.resize_token_embeddings(len(tokenizer))

    #Inspect vision model
    print("Model loaded successfully.")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Processor architecture: {processor.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Inspect model architecture to find correct module names
    print("\nInspecting model architecture...")
    print("First few module names:")
    module_names = []
    for name, module in model.named_modules():
        module_names.append(name)
        if len(module_names) <= 20:  # Print first 20 module names
            print(f"  {name}: {module.__class__.__name__}")
    
    # Look for attention and MLP modules specifically
    print("\nLooking for attention and MLP modules...")
    attention_modules = []
    mlp_modules = []
    for name in module_names:
        if any(keyword in name.lower() for keyword in ['attn', 'attention']):
            attention_modules.append(name)
        if any(keyword in name.lower() for keyword in ['mlp', 'feed_forward', 'ffn']):
            mlp_modules.append(name)
    
    print(f"Found {len(attention_modules)} attention modules")
    print(f"Found {len(mlp_modules)} MLP modules")
    if attention_modules:
        print("Sample attention modules:", attention_modules[:5])
    if mlp_modules:
        print("Sample MLP modules:", mlp_modules[:5])

    # Simplified LoRA configuration using actual module names found in the model
    # Start with basic attention modules that are commonly present
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Standard attention projections
        "gate_proj", "up_proj", "down_proj"      # Standard MLP projections
    ]
    
    # Try to find actual module names that contain these patterns
    actual_target_modules = []
    for name, _ in model.named_modules():
        for target in target_modules:
            if target in name and name not in actual_target_modules:
                actual_target_modules.append(name)
                break
    
    # If we found modules, use them; otherwise fall back to common patterns
    if actual_target_modules:
        print(f"Found {len(actual_target_modules)} target modules for LoRA:")
        for module in actual_target_modules[:10]:  # Show first 10
            print(f"  {module}")
        final_target_modules = actual_target_modules
    else:
        print("No specific modules found, using pattern matching...")
        final_target_modules = [
            "*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj",
            "*.gate_proj", "*.up_proj", "*.down_proj"
        ]

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=final_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=False,
        use_dora=False,
    )

    try:
        model = get_peft_model(model, lora_cfg)
        print("LoRA adapter successfully applied!")
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        print("Trying with even simpler target modules...")
        
        # Fallback to the most basic configuration
        simple_target_modules = ["q_proj", "v_proj"]
        lora_cfg_simple = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=simple_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=False,
            use_dora=False,
        )
        model = get_peft_model(model, lora_cfg_simple)
        print("LoRA adapter applied with simplified configuration!")

    # ---------------------------------------------------------------------
    # 2.  LOAD + PREP DATA  ────────────────────────────────────────────────
    # ---------------------------------------------------------------------
    print("Loading and preparing dataset...")
    raw_ds_eval = load_dataset("danjacobellis/chexpert", "default", split="validation")

    if args.n_samples_train > 0:
        # Load full training set and select subset
        raw_ds_train_full = load_dataset("danjacobellis/chexpert", "default", split="train")
        available_samples = len(raw_ds_train_full)
        actual_n_samples = min(args.n_samples_train, available_samples)
        print(f"Using subset of {actual_n_samples} samples for training (requested: {args.n_samples_train}, available: {available_samples})")
        raw_ds_train = raw_ds_train_full.shuffle(seed=42).select(range(actual_n_samples))
    else:
        print("Using full CheXpert 'train' split for training")
        raw_ds_train = load_dataset("danjacobellis/chexpert", "default", split="train")

    # Prepare instruction text
    instruction_text = (
    
    "In <thinking>:\n"
    "Systematically examine each anatomical region in the image (heart, lungs, pleura, mediastinum, bones, devices)\n"
    "Describe what you see in detail in the image (size, shape, density, position of structures)\n"
    "Identify any abnormalities and explain their medical significance\n"
    "Consider possible diagnoses and rule them in or out based on evidence\n"
    "If uncertain about any finding, re-examine that area in teh image more carefully. Describe all your reasoning step by step. \n\n"
    
    "In <answer>:\n\n"
    f"List only the applicable conditions from: {', '.join(LABEL_COLS)}\n\n"
    "Use exact label names, separated by commas.\n\n"
    "Answer format:\n"
    "<thinking>your thinking here ...</thinking>"
    "<answer>Your answer here ...</answer>"
    "Your response must follow this exact structure without any extra text before <thinking> or after </answer>"
    "EXAMPLE:\n"
    "<thinking>I examine the cardiac silhouette which appears enlarged with a cardiothoracic ratio exceeding 50%. "
    "The left heart border is prominent suggesting left ventricular enlargement. The lung fields show bilateral "
    "lower lobe opacities with air bronchograms consistent with consolidation. The costophrenic angles are "
    "blunted bilaterally indicating pleural effusions. No pneumothorax is visible. The mediastinum is not "
    "widened. Based on these findings, I identify cardiomegaly, lung consolidation suggesting pneumonia, "
    "and bilateral pleural effusions.</thinking>\n"
    "<answer>Cardiomegaly, Pneumonia, Pleural Effusion</answer>\n\n"
    
    "Your response must follow this exact structure without any extra text before <thinking> or after </answer>"
    )
    
    # Prepare datasets with prompts and ground truth labels
    prepare_prompt_with_processor_and_instruction = lambda sample: prepare_prompt(sample, processor, instruction_text)

    proc_ds_train = (
        raw_ds_train.map(extract_gt, remove_columns=[])
                    .map(prepare_prompt_with_processor_and_instruction, remove_columns=[])
    )
    proc_ds_eval = (
        raw_ds_eval.map(extract_gt, remove_columns=[])
                   .map(prepare_prompt_with_processor_and_instruction, remove_columns=[])
    )
    
    print(f"Training dataset size: {len(proc_ds_train)}")
    print(f"Evaluation dataset size: {len(proc_ds_eval)}")

    # ---------------------------------------------------------------------
    # 4.  CONFIG + TRAIN  ─────────────────────────────────────────────────
    # ---------------------------------------------------------------------
    print("Configuring GRPOTrainer...")
    config = GRPOConfig(
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        sync_ref_model=True,
        logging_steps=1,
        max_completion_length=256,  # Reduced for stability
        max_prompt_length=256,      # Reduced for stability
        temperature=0.6,            # Reduced for stability
        fp16=False,
        bf16=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        group_by_length=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="tensorboard",
        output_dir=checkpoint_dir,
        logging_dir=tensorboard_path,
        save_total_limit=1,
        gradient_checkpointing=False,  # Enable for memory efficiency
        # Disable vLLM for now to avoid compatibility issues
        #use_vllm=True,
        #vllm_mode="colocate"
    )

    # Create trainer with proper error handling



    
    try:
        trainer = Qwen2VLGRPOTrainer(
            model=model,
            reward_funcs=[
                reward_format,
                reward_classification,
            ],
            args=config,
            train_dataset=proc_ds_train,
            eval_dataset=proc_ds_eval,
            processing_class=processor,
        )
    except Exception as e:
        print(f"Error creating GRPOTrainer: {e}")
        print("Falling back to standard training approach...")
        # Add fallback training logic here if needed
        raise e

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # ---------------------------------------------------------------------
    # 5.  SAVE FINAL MODEL  ──────────────────────────────────────────────
    # ---------------------------------------------------------------------
    print(f"Saving final LoRA adapter to {final_model_path}")
    model.save_lora(final_model_path)
    print(f"Final LoRA adapter saved to {final_model_path}")
    print(f"Final LoRA adapter saved to {final_model_path}")
    
if __name__ == "__main__":
    main()
