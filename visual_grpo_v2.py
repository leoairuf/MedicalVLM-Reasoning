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
import re, torch, os
import argparse
from collections import defaultdict
import glob

# Set Hugging Face cache to current working directory
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir

from datasets import load_dataset, config
print(f"Cache directory: {config.HF_DATASETS_CACHE}")
from sklearn.metrics import f1_score

from unsloth import FastModel, FastVisionModel
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

import types, unsloth_zoo.peft_utils as _pz

# ---------------------------------------------------------------------
# 0.  GLOBAL HYPERPARAMETERS & CONFIGURATIONS  ─────────────────────────
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME = "unsloth/gemma-3-4b-it"
DEFAULT_N_SAMPLES_TRAIN = 5000  # Number of samples for quick testing
DEFAULT_MAX_STEPS = 5000
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1  # 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_OUTPUT_DIR = "outputs_v2"
DEFAULT_TENSORBOARD_DIR = "tensorboard"
DEFAULT_SAVE_STEPS = 50
DEFAULT_RESUME_FROM_CHECKPOINT = "auto"  # Path to checkpoint to resume from, or 'auto' to find latest automatically

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

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory in the output directory"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Extract step numbers and find the highest one
    checkpoint_steps = []
    for cp in checkpoints:
        try:
            step = int(os.path.basename(cp).split('-')[1])
            checkpoint_steps.append((step, cp))
        except (IndexError, ValueError):
            continue
    
    if not checkpoint_steps:
        return None
    
    # Return the checkpoint with the highest step number
    latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])[1]
    return latest_checkpoint

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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from. If 'auto', will find latest checkpoint automatically.")
    
    return parser.parse_args()

# ---------------------------------------------------------------------
# Helper functions for prompt preparation and label extraction
# ---------------------------------------------------------------------

def prepare_prompt(sample, tokenizer_ref, instruction_text):
    """Build VLM chat prompt compatible with unsloth/Qwen template"""
    conversation = [
        {   
            #"role": "system", "content": SYSTEM_PROMPT,
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": instruction_text},
            ],
        }
    ]
    sample["prompt"] = tokenizer_ref.apply_chat_template(
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
    rewards = []
    for idx, text in enumerate(completions):
        if structure_regex.search(text) and answer_regex.search(text).group(1).strip():
            r = 1.0
        elif structure_regex.search(text):
            r = -0.5
        else:
            r = -1.0
        print(f"[Format] idx={idx} reward={r}")
        print(f"[Format] idx={idx} Generated Text:\n{text}")
        rewards.append(r)
    return rewards


def reward_classification(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Scaled sample-level F1 reward: 2*F1(samples)-1, where F1 is computed with average='samples'.
    """
    gt_list = kwargs.get("gt_labels", [])
    rewards = []
    for idx, (text, gt_labels) in enumerate(zip(completions, gt_list)):
        pred_labels = parse_pred(text)
        y_true = [1 if lbl in gt_labels else 0 for lbl in LABEL_COLS]
        y_pred = [1 if lbl in pred_labels else 0 for lbl in LABEL_COLS]
        f1_sample = f1_score([y_true], [y_pred], average="samples", zero_division=0)
        r = 2 * f1_sample - 1
        print(f"[Class] idx={idx} GT={gt_labels} PRED={pred_labels} f1={f1_sample:.3f} reward={r:.3f}")
        rewards.append(r)
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

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    tensorboard_path = os.path.join(args.output_dir, args.tensorboard_dir)
    os.makedirs(tensorboard_path, exist_ok=True)

    print(f"Output directory: {args.output_dir}")
    print(f"Tensorboard logs: {tensorboard_path}")

    # Check for checkpoint resume
    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "auto":
            resume_from_checkpoint = find_latest_checkpoint(args.output_dir)
            if resume_from_checkpoint:
                print(f"Auto-detected checkpoint: {resume_from_checkpoint}")
            else:
                print("No checkpoint found for auto-resume, starting from scratch")
        else:
            if os.path.exists(args.resume_from_checkpoint):
                resume_from_checkpoint = args.resume_from_checkpoint
                print(f"Resuming from specified checkpoint: {resume_from_checkpoint}")
            else:
                print(f"Warning: Specified checkpoint {args.resume_from_checkpoint} not found, starting from scratch")

    # ---------------------------------------------------------------------
    # 1.  LOAD MODEL ─────────────────────────────────────────────────────
    # ---------------------------------------------------------------------
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        load_in_8bit=False,
        #fast_inference = True,
        #gpu_memory_utilization = 0.6,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        #bias="none",
        random_state=3407,
    )

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
    instruction_text1 = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <thinking> </thinking> and <answer> </answer> tags, respectively, i.e., "
    "<thinking> reasoning process here </thinking><answer> answer here </answer>. \n\n"
    "You are an expert radiologist specializing in chest X-ray interpretation.\n\n"
    "Your task:\n"
    " 1. Systematically analyze the image, step by step.\n"
    "    In each step of your reasoning, clearly identify the anatomical region you are examining, explain why you focus on it, and describe the medical significance of your findings. Locate also the region in the chest X-ray image whith cordinates. \n"
    " 2. When your analysis is complete, list only the detected conditions, chosen from: " + ", ".join(LABEL_COLS) + ".\n\n"
    "Required output format:\n"
    "<thinking>\n"
    "Your detailed visual reasoning, step by step, with anatomical localization and medical rationale\n"
    "</thinking>\n"
    "<answer>\n"
    "  [Exact label names of the detected conditions, comma-separated]\n"
    "</answer>\n\n"
    "EXAMPLE:\n"
    "<thinking>I examine the cardiac silhouette which appears enlarged with a cardiothoracic ratio exceeding 50%. "
    "The left heart border is prominent suggesting left ventricular enlargement. The lung fields show bilateral "
    "lower lobe opacities with air bronchograms consistent with consolidation. The costophrenic angles are "
    "blunted bilaterally indicating pleural effusions. No pneumothorax is visible. The mediastinum is not "
    "widened. Based on these findings, I identify cardiomegaly, lung consolidation suggesting pneumonia, "
    "and bilateral pleural effusions.</thinking>"
    "<answer>Cardiomegaly, Pneumonia, Pleural Effusion</answer>"

    "Your response must follow this exact structure without any extra text before <thinking> or after </answer>."
    )

    instruction_text = (
    "You are a radiologist analyzing a chest X-ray. Provide your complete diagnostic reasoning step by step in <thinking> tags, "
    "then your final diagnosis in <answer> tags.\n\n"
    
    "In <thinking>:\n"
    "Systematically examine each anatomical region in the image (heart, lungs, pleura, mediastinum, bones, devices)\n"
    "Describe what you see in detail in the image (size, shape, density, position of structures)\n"
    "Identify any abnormalities and explain their medical significance\n"
    "Consider possible diagnoses and rule them in or out based on evidence\n"
    "If uncertain about any finding, re-examine that area in teh image more carefully. Describe all your reasoning step by step. \n\n"
    
    "In <answer>:\n\n"
    f"List only the applicable conditions from: {', '.join(LABEL_COLS)}\n\n"
    "Use exact label names, separated by commas.\n\n"

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
    prepare_prompt_with_tokenizer_and_instruction = lambda sample: prepare_prompt(sample, tokenizer, instruction_text)

    proc_ds_train = (
        raw_ds_train.map(extract_gt, remove_columns=[])
                    .map(prepare_prompt_with_tokenizer_and_instruction, remove_columns=[])
    )
    proc_ds_eval = (
        raw_ds_eval.map(extract_gt, remove_columns=[])
                   .map(prepare_prompt_with_tokenizer_and_instruction, remove_columns=[])
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
        optim="paged_adamw_8bit",
        sync_ref_model=True,
        logging_steps=1,
        max_completion_length=2048,
        temperature=1.1,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="tensorboard",
        output_dir=args.output_dir,
        logging_dir=tensorboard_path,
        save_total_limit=1,
        resume_from_checkpoint=resume_from_checkpoint,  # Add this line
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_format,
            reward_classification,
            reward_exact_match,
        ],
        args=config,
        train_dataset=proc_ds_train,
        eval_dataset=proc_ds_eval,
        processing_class=tokenizer,
    )

    print("Starting training...")
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Training finished.")

    # ---------------------------------------------------------------------
    # 5.  SAVE FINAL MODEL  ──────────────────────────────────────────────
    # ---------------------------------------------------------------------
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final LoRA adapter to {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final LoRA adapter and tokenizer saved to {final_model_path}")

if __name__ == "__main__":
    main()
