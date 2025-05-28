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
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_OUTPUT_DIR = "outputs"
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

def prepare_prompt(sample, tokenizer_ref, instruction_text):
    """Build VLM chat prompt compatible with unsloth/Qwen template"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_text},
                {"type": "image", "image": sample["image"]},
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

def parse_pred(text: str):
    """Extract text within <answer> tags and return unique list of category names found"""
    answer_match = answer_regex.search(text)
    if not answer_match:
        return []  # No answer tag found
    answer_content = answer_match.group(1).strip()
    # Find valid labels within the answer content
    return list({m.group(0).title() for m in label_regex.finditer(answer_content)})

def format_reward(pred_text: str) -> float:
    """Return +1.0 if output matches <thinking>...</thinking><answer>...</answer> structure, -1.0 otherwise"""
    if structure_regex.search(pred_text):
        # Check if answer tag is not empty
        answer_match = answer_regex.search(pred_text)
        if answer_match and answer_match.group(1).strip():
            return 1.0  # Exact format match and non-empty answer
        else:
            return -0.5  # Correct structure but empty answer tag
    return -1.0  # Incorrect structure

def classification_reward(pred_text: str, gt_labels) -> float:
    """Calculate reward based on F1 score of labels extracted from <answer> tag"""
    pred = parse_pred(pred_text)
    y_true = [1 if lbl in gt_labels else 0 for lbl in LABEL_COLS]
    y_pred = [1 if lbl in pred else 0 for lbl in LABEL_COLS]
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 0.5  # Neutral when both empty
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return 2 * f1 - 1  # Scale F1∈[0,1] → reward∈[-1,1]

def reward_fn(prompts: list[str], completions: list[str], **kwargs):
    """
    Calculate rewards for each completion based on format and classification F1 score.
    Assumes 1 completion per prompt.

    Args:
        prompts (list[str]): Input prompts (length == original batch size)
        completions (list[str]): Generated completions (expected length == original batch size)
        **kwargs: Additional arguments, containing 'gt_labels' as a list

    Returns:
        list[float]: Rewards for each completion
    """
    gt_labels_list = kwargs.get("gt_labels")
    original_batch_size = len(prompts)
    received_completions_count = len(completions)

    # Validate gt_labels
    if not isinstance(gt_labels_list, list) or len(gt_labels_list) != original_batch_size:
        print("Error: 'gt_labels' not found in kwargs or does not match original batch size.")
        return [-1.0] * received_completions_count

    # Check completion count
    if received_completions_count != original_batch_size:
        print(f"Warning: Received {received_completions_count} completions, expected {original_batch_size}")

    rewards = []
    num_rewards_to_calculate = min(received_completions_count, original_batch_size)

    for idx in range(num_rewards_to_calculate):
        pred_text = completions[idx]
        current_gt_labels = gt_labels_list[idx]

        # Calculate format reward
        fmt_r = format_reward(pred_text)
        
        # Calculate classification reward
        cls_r = -1.0  # Default in case of error
        pred_labels = []
        try:
            pred_labels = parse_pred(pred_text)
            y_true = [1 if lbl in current_gt_labels else 0 for lbl in LABEL_COLS]
            y_pred = [1 if lbl in pred_labels else 0 for lbl in LABEL_COLS]
            if sum(y_true) == 0 and sum(y_pred) == 0:
                cls_r = 0.5
            else:
                f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
                cls_r = 2 * f1 - 1
        except Exception as e:
            print(f"Error in classification_reward for completion {idx}: {e}")

        # Combine rewards (50% format, 50% classification)
        final_reward = 0.5 * fmt_r + 0.5 * cls_r
        
        # Bonus for exact label match
        if set(pred_labels) == set(current_gt_labels):
            final_reward += 1.0

        rewards.append(final_reward)

        # # Debug prints (commented out for production)
        #print(f"\n--- Sample {idx} ---")
        #print(f"Generated Text:\n{pred_text}")
        #print(f"Ground Truth Labels: {current_gt_labels}")
        #print(f"Format Reward: {fmt_r}")
        #print(f"Parsed Predicted Labels: {pred_labels}")
        #print(f"Classification Reward: {cls_r}")
        #if set(pred_labels) == set(current_gt_labels):
        #    print(f"Exact match bonus +1.0 applied")
        #print(f"Final Reward: {final_reward}")

    # Ensure reward count matches completion count
    if len(rewards) != received_completions_count:
        print(f"Warning: Adjusting reward count from {len(rewards)} to {received_completions_count}")
        rewards.extend([-1.0] * (received_completions_count - len(rewards)))
        rewards = rewards[:received_completions_count]

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

    # ---------------------------------------------------------------------
    # 1.  LOAD MODEL ─────────────────────────────────────────────────────
    # ---------------------------------------------------------------------
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        load_in_8bit=False,
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
    instruction_text = (
        "You are a radiologist analyzing a chest X-ray. Provide your complete diagnostic reasoning step by step in <thinking> tags, "
        "then your final diagnosis in <answer> tags.\n\n"
        
        "In <thinking>:\n"
        "Systematically examine each anatomical region (heart, lungs, pleura, mediastinum, bones, devices)\n"
        "Describe what you see in detail (size, shape, density, position of structures)\n"
        "Identify any abnormalities and explain their medical significance\n"
        "Consider possible diagnoses and rule them in or out based on evidence\n"
        "If uncertain about any finding, re-examine that area more carefully. Describe all your reasoning step by step. \n\n"
        
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
        
        "Your response must follow this exact structure without any extra text before <thinking> or after </answer>."
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
        max_completion_length=3000,
        temperature=1.1,
        #evaluation_strategy="steps",
        #eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="tensorboard",
        output_dir=args.output_dir,
        logging_dir=tensorboard_path,
        save_total_limit=1,  # Keep only the latest checkpoint to save disk space
        #log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=proc_ds_train,
        eval_dataset=proc_ds_eval,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # ---------------------------------------------------------------------
    # 5.  SAVE FINAL MODEL  ──────────────────────────────────────────────
    # ---------------------------------------------------------------------
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final LoRA adapter to {final_model_path}")
    model.save_lora(final_model_path)
    print(f"Final LoRA adapter saved to {final_model_path}")

if __name__ == "__main__":
    main()
