# -*- coding: utf-8 -*-
"""
GRPO fine‑tuning of Qwen2.5‑VL‑3B on CheXpert multi‑label classification
-----------------------------------------------------------------------
This script continues from the user‑supplied cells and adds:
  • dataset preprocessing (ground‑truth label extraction + prompt building)
  • two reward functions (format + classification)
  • a composite reward_fn compatible with Unsloth GRPOTrainer
  • GRPO training loop and LoRA export

Run in environment with: pip install unsloth "scikit-learn>=1.4" datasets
"""
import re, torch, os
from collections import defaultdict
from datasets import load_dataset
from sklearn.metrics import f1_score

from unsloth import FastModel, FastVisionModel

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer


import types, unsloth_zoo.peft_utils as _pz


# ---------------------------------------------------------------------
# 1.  LOAD MODEL (continues previous code)  ────────────────────────────
# ---------------------------------------------------------------------
model_name = "unsloth/gemma-3-4b-it"
model, tokenizer = FastVisionModel.from_pretrained(
    model_name,
    load_in_4bit=True,
    use_gradient_checkpointing = "unsloth",  # disabilitato per evitare il hook PEFT
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
)

# Instantiate a text-only tokenizer to use with GRPOTrainer
#text_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# ---------------------------------------------------------------------
# 2.  LOAD + PREP DATA  ────────────────────────────────────────────────
# ---------------------------------------------------------------------
label_cols = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

# Load train and validation splits
raw_ds_train = load_dataset("danjacobellis/chexpert", "default", split="validation")   # set to "train" for training
raw_ds_eval = load_dataset("danjacobellis/chexpert", "default", split="validation")

# Updated instruction for structured reasoning and answer
# Added emphasis on ONLY outputting the tags.
instruction = (
    "Analyze the provided chest X-ray image. First, provide your step-by-step reasoning "
    "for the diagnosis within <thinking> tags. Then, provide the final classification "
    "labels within <answer> tags. The classification should be a comma-separated list "
    f"of applicable labels from the following: {', '.join(label_cols)}. "
    "Example format: <thinking>The heart appears enlarged...</thinking><answer>Cardiomegaly, Pleural Effusion</answer>"
    "IMPORTANT: Your entire response must contain ONLY the <thinking>...</thinking><answer>...</answer> structure. "
    "Do not include any text before the <thinking> tag or after the </answer> tag."
)

# helper: build VLM chat prompt compatible with unsloth/Qwen template

def prepare_prompt(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        }
    ]
    # Removed print(f"Conversation: {conversation}") to avoid excessive output
    #sample["messages"] = conversation
    # Add the formatted prompt string explicitly
    sample["prompt"] = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True, # Important for generation
    )
    return sample

# add ground‑truth label list

def extract_gt(sample):
    # Correctly identify 'present' labels (value 3)
    # Uncertain (1) and Absent (2) are treated as not present for binary classification reward
    sample["gt_labels"] = [c for c in label_cols if sample[c] == 3]
    return sample

# Process both datasets
proc_ds_train = (
    raw_ds_train.map(extract_gt, remove_columns=[])
                .map(prepare_prompt, remove_columns=[])  # keep all cols for rewards
)
proc_ds_eval = (
    raw_ds_eval.map(extract_gt, remove_columns=[])
               .map(prepare_prompt, remove_columns=[])  # keep all cols for rewards
)

# ---------------------------------------------------------------------
# 3.  REWARD FUNCTIONS  ────────────────────────────────────────────────
# Regex to find labels (case-insensitive)
label_regex = re.compile(
    r"|".join([re.escape(c) for c in label_cols]), re.IGNORECASE
)
# Regex to extract content within <answer> tags
answer_regex = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
# Regex to check for the overall structure - updated for exact match
# Allows optional leading/trailing whitespace but nothing else outside the tags.
structure_regex = re.compile(r"^\s*<thinking>.*?</thinking>\s*<answer>.*?</answer>\s*$", re.IGNORECASE | re.DOTALL)


def parse_pred(text: str):
    """Extracts the text within <answer> tags and returns a unique list of category names found."""
    answer_match = answer_regex.search(text)
    if not answer_match:
        return []  # No answer tag found
    answer_content = answer_match.group(1).strip()
    # Find valid labels within the answer content
    return list({m.group(0).title() for m in label_regex.finditer(answer_content)})


def format_reward(pred_text: str) -> float:
    """+1.0 if the output EXACTLY matches the <thinking>...</thinking><answer>...</answer> structure, -1.0 otherwise."""
    # Check if the entire string matches the required structure exactly.
    if structure_regex.search(pred_text):
        # Further check: ensure answer tag is not empty (optional, but good practice)
        answer_match = answer_regex.search(pred_text)
        if answer_match and answer_match.group(1).strip():
            return 1.0 # Exact format match and non-empty answer
        else:
            # Matched structure but empty answer tag - penalize less severely than wrong structure?
            # Or treat as format failure? Let's treat as format failure for simplicity.
            return -0.5 # Penalize empty answer tag within correct structure
    # If the regex doesn't match the entire string structure
    return -1.0 # Penalize incorrect structure or extra text


def classification_reward(pred_text: str, gt_labels) -> float:
    """Calculates reward based on F1 score of labels extracted from <answer> tag."""
    pred = parse_pred(pred_text) # Uses the updated parse_pred
    y_true = [1 if lbl in gt_labels else 0 for lbl in label_cols]
    y_pred = [1 if lbl in pred else 0 for lbl in label_cols]
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 0.5  # neutral when both empty
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0) # Added zero_division=0
    return 2 * f1 - 1  # scale F1∈[0,1] → reward∈[-1,1]


# GRPOTrainer expects a single callable that returns list[float]
# Updated signature to accept arguments passed by GRPOTrainer
# Handles the case where only 1 completion per prompt is received
def reward_fn(prompts: list[str], completions: list[str], **kwargs):
    """
    Calculates rewards for each completion based on format and classification F1 score.
    Assumes 1 completion is passed per original prompt, despite config.num_generations.

    Args:
        prompts (list[str]): The list of input prompts (length == original batch size).
        completions (list[str]): The list of generated completions (expected length == original batch size).
        **kwargs: Additional keyword arguments, containing 'gt_labels' as a list.

    Returns:
        list[float]: A flat list of rewards, one for each completion received.
    """
    gt_labels_list = kwargs.get("gt_labels")
    original_batch_size = len(prompts)
    received_completions_count = len(completions)

    # Check if gt_labels_list is a list matching the original batch size
    if not isinstance(gt_labels_list, list) or len(gt_labels_list) != original_batch_size:
        # --- Failure Case: gt_labels is missing, not a list, or wrong length ---
        print("Error: 'gt_labels' not found in kwargs or does not match original batch size.")
        # ... (optional debug prints as before) ...
        return [-1.0] * received_completions_count # Return default low rewards

    # Check if the number of completions matches the number of prompts (our new expectation)
    if received_completions_count != original_batch_size:
        print(f"Warning: Received {received_completions_count} completions, but expected {original_batch_size} (one per prompt).")
        # Decide how to handle: pad rewards, raise error, or process what's received?
        # Let's process what we received and pad/truncate rewards at the end if needed.
        pass # Continue processing

    rewards = []
    # Iterate up to the minimum of received completions and original batch size
    num_rewards_to_calculate = min(received_completions_count, original_batch_size)

    for idx in range(num_rewards_to_calculate):
        pred_text = completions[idx]
        current_gt_labels = gt_labels_list[idx] # Get GT for this original sample

        # --- Debug Prints Start ---
        print(f"\n--- Sample {idx} ---")
        print(f"Generated Text:\n{pred_text}")
        print(f"Ground Truth Labels: {current_gt_labels}")

        # Use the updated format_reward
        fmt_r = format_reward(pred_text)
        print(f"Format Reward (fmt_r): {fmt_r}") # Stampa il risultato di format_reward

        cls_r = -1.0 # Default in caso di errore
        # Only calculate classification reward if format is somewhat correct (contains answer tag)
        # to avoid errors in parse_pred if the format is completely wrong.
        # Alternatively, always calculate it, as parse_pred handles missing tags.
        # Let's keep calculating it, as parse_pred returns [] if no answer tag.
        try:
            # Stampa i risultati intermedi di classification_reward
            pred_labels = parse_pred(pred_text)
            print(f"Parsed Predicted Labels: {pred_labels}")
            y_true = [1 if lbl in current_gt_labels else 0 for lbl in label_cols]
            y_pred = [1 if lbl in pred_labels else 0 for lbl in label_cols]
            print(f"y_true (binary): {y_true}")
            print(f"y_pred (binary): {y_pred}")
            if sum(y_true) == 0 and sum(y_pred) == 0:
                cls_r = 0.5
                print("Classification Reward (cls_r): 0.5 (both GT and Pred empty)")
            else:
                f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
                cls_r = 2 * f1 - 1
                print(f"Micro F1 Score: {f1}")
                print(f"Classification Reward (cls_r): {cls_r}") # Stampa il risultato di classification_reward
        except Exception as e:
            print(f"Error in classification_reward for completion {idx}: {e}")
            print(f"Classification Reward (cls_r): {cls_r} (due to error)")

        # Adjust weighting? Maybe give more weight to format now?
        # Example: 50% format, 50% classification
        final_reward = 0.5 * fmt_r + 0.5 * cls_r
        rewards.append(final_reward)
        print(f"Combined Reward (50/50): {final_reward}") # Updated weighting
        print(f"--- End Sample {idx} ---")
        # --- Debug Prints End ---

    # Ensure the number of rewards matches the number of completions *received*
    if len(rewards) != received_completions_count:
         print(f"Warning: Final reward count ({len(rewards)}) does not match received completions ({received_completions_count}). Padding/Truncating.")
         rewards.extend([-1.0] * (received_completions_count - len(rewards)))
         rewards = rewards[:received_completions_count]

    return rewards

# ---------------------------------------------------------------------
# 4.  CONFIG + TRAIN  ─────────────────────────────────────────────────
# ---------------------------------------------------------------------
config = GRPOConfig(
    max_steps=500,          # increase for better results
    learning_rate=5e-6,
    num_generations=8,      # completions per prompt
    per_device_train_batch_size=8,      # aggiornato da 2 a 8
    per_device_eval_batch_size=8,       # coerente con train
    gradient_accumulation_steps=2,      # per mantenere batch totali
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    sync_ref_model = True, # Sync ref model with main model after each step
    logging_steps=1, # Log less frequently
    #evaluation_strategy="steps", # Enable evaluation during training
    #eval_steps=50, # Evaluate every 50 steps
    #save_strategy="steps", # Save checkpoints based on steps
    #save_steps=50, # Save every 50 steps (aligned with eval)
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model=model,
    # Pass the updated reward_fn directly
    reward_funcs=reward_fn,
    args=config,
    train_dataset=proc_ds_train, # Use processed training data
    eval_dataset=proc_ds_eval,   # Use processed validation data
    # Ensure the tokenizer is passed if needed by the trainer for processing completions/prompts internally
    # If the multi-modal tokenizer works for text, it might be fine.
    # Otherwise, consider using a text-only tokenizer here if available and appropriate.
    processing_class=tokenizer, # Changed from commented out to active
)

trainer.train()

# ---------------------------------------------------------------------
# 5.  SAVE LoRA ADAPTER  ──────────────────────────────────────────────
# ---------------------------------------------------------------------
model.save_lora("chexpert_grpo_lora")
print("Training finished – LoRA adapter saved to ./chexpert_grpo_lora")
