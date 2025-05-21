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
import argparse # Aggiunto argparse
from collections import defaultdict
from datasets import load_dataset
from sklearn.metrics import f1_score

# Suggerimento: per l'errore "Too many open files", prova ad aumentare il limite
# di file aperti nel tuo terminale prima di eseguire lo script, ad esempio:
# ulimit -n 65535
# Imposta WANDB_DISABLED per evitare problemi con i file aperti se wandb non è in uso.
os.environ["WANDB_DISABLED"] = "false" # Modificato per abilitare wandb

from unsloth import FastModel, FastVisionModel

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

import wandb # Aggiunto import wandb

import types, unsloth_zoo.peft_utils as _pz

# ---------------------------------------------------------------------
# 0.  GLOBAL HYPERPARAMETERS & CONFIGURATIONS  ─────────────────────────
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME = "unsloth/gemma-3-4b-it"
DEFAULT_N_SAMPLES_TRAIN = 3  # Numero di campioni per il test rapido
DEFAULT_MAX_STEPS = 500
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_OUTPUT_LORA_PATH = "chexpert_grpo_lora"
DEFAULT_REPORT_TO = "wandb" # o "none" se non si usa wandb

# Etichette per la classificazione CheXpert
LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]

# Regex globali (definite una volta)
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
    parser.add_argument("--output_lora_path", type=str, default=DEFAULT_OUTPUT_LORA_PATH, help="Path to save the LoRA adapter.")
    parser.add_argument("--report_to", type=str, default=DEFAULT_REPORT_TO, help="Integration for reporting metrics (e.g., 'wandb', 'none').")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable WANDB logging explicitly.")


    return parser.parse_args()

# ---------------------------------------------------------------------
# Funzioni di supporto (prepare_prompt, extract_gt, parse_pred, format_reward, classification_reward, reward_fn)
# Rimangono per lo più invariate, ma useranno LABEL_COLS, label_regex, ecc. definite globalmente.
# ---------------------------------------------------------------------

# helper: build VLM chat prompt compatible with unsloth/Qwen template

def prepare_prompt(sample, tokenizer_ref, instruction_text): # Aggiunto instruction_text
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_text}, # Usa instruction_text
                {"type": "image", "image": sample["image"]},
            ],
        }
    ]
    sample["prompt"] = tokenizer_ref.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True, # Important for generation
    )
    return sample

# add ground‑truth label list

def extract_gt(sample):
    # Correctly identify 'present' labels (value 3)
    # Uncertain (1) and Absent (2) are treated as not present for binary classification reward
    sample["gt_labels"] = [c for c in LABEL_COLS if sample[c] == 3] # Usa LABEL_COLS
    return sample

# Process both datasets
# Questa logica sarà dentro main()

# ---------------------------------------------------------------------
# 3.  REWARD FUNCTIONS  ────────────────────────────────────────────────
# Regex sono già definite globalmente

def parse_pred(text: str):
    """Extracts the text within <answer> tags and returns a unique list of category names found."""
    answer_match = answer_regex.search(text) # Usa answer_regex globale
    if not answer_match:
        return []  # No answer tag found
    answer_content = answer_match.group(1).strip()
    # Find valid labels within the answer content
    return list({m.group(0).title() for m in label_regex.finditer(answer_content)}) # Usa label_regex globale


def format_reward(pred_text: str) -> float:
    """+1.0 if the output EXACTLY matches the <thinking>...</thinking><answer>...</answer> structure, -1.0 otherwise."""
    # Check if the entire string matches the required structure exactly.
    if structure_regex.search(pred_text): # Usa structure_regex globale
        # Further check: ensure answer tag is not empty (optional, but good practice)
        answer_match = answer_regex.search(pred_text) # Usa answer_regex globale
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
    y_true = [1 if lbl in gt_labels else 0 for lbl in LABEL_COLS] # Usa LABEL_COLS
    y_pred = [1 if lbl in pred else 0 for lbl in LABEL_COLS] # Usa LABEL_COLS
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
        pred_labels = [] # Inizializza pred_labels
        # Only calculate classification reward if format is somewhat correct (contains answer tag)
        # to avoid errors in parse_pred if the format is completely wrong.
        # Alternatively, always calculate it, as parse_pred handles missing tags.
        # Let's keep calculating it, as parse_pred returns [] if no answer tag.
        try:
            # Stampa i risultati intermedi di classification_reward
            pred_labels = parse_pred(pred_text) # pred_labels viene assegnato qui
            print(f"Parsed Predicted Labels: {pred_labels}")
            y_true = [1 if lbl in current_gt_labels else 0 for lbl in LABEL_COLS] # Usa LABEL_COLS
            y_pred = [1 if lbl in pred_labels else 0 for lbl in LABEL_COLS] # Usa LABEL_COLS
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
        
        # Aggiungi bonus se la lista delle etichette predette è ESATTAMENTE corretta
        # Confronta come insiemi per ignorare l'ordine e gestire l'unicità
        # pred_labels è già stato calcolato sopra
        if set(pred_labels) == set(current_gt_labels):
            final_reward += 1.0
            print(f"Bonus +1.0 applicato per corrispondenza esatta delle etichette.")

        rewards.append(final_reward)
        print(f"Combined Reward (50/50, con bonus potenziale): {final_reward}") # Updated weighting and message
        print(f"--- End Sample {idx} ---")
        # --- Debug Prints End ---

    # Ensure the number of rewards matches the number of completions *received*
    if len(rewards) != received_completions_count:
         print(f"Warning: Final reward count ({len(rewards)}) does not match received completions ({received_completions_count}). Padding/Truncating.")
         rewards.extend([-1.0] * (received_completions_count - len(rewards)))
         rewards = rewards[:received_completions_count]

    return rewards

# ---------------------------------------------------------------------
# MAIN SCRIPT LOGIC  ──────────────────────────────────────────────────
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    elif args.report_to == "wandb":
        os.environ["WANDB_DISABLED"] = "false"
        wandb.init(project="MedicalVLM-Reasoning-GRPO", config=args) # Inizializza wandb
    else:
        os.environ["WANDB_DISABLED"] = "true"


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
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
    )

    # ---------------------------------------------------------------------
    # 2.  LOAD + PREP DATA  ────────────────────────────────────────────────
    # ---------------------------------------------------------------------
    print("Loading and preparing dataset...")
    # raw_ds_train = load_dataset("danjacobellis/chexpert", "default", split="validation") # Original validation as train
    raw_ds_eval = load_dataset("danjacobellis/chexpert", "default", split="validation")

    if args.n_samples_train > 0:
        print(f"Using a subset of {args.n_samples_train} samples for training (from validation set).")
        raw_ds_train = raw_ds_eval.shuffle(seed=42).select(range(args.n_samples_train))
    else:
        print("Using full CheXpert 'train' split for training.")
        raw_ds_train = load_dataset("danjacobellis/chexpert", "default", split="train")


    instruction_text = (
        "Analyze the provided chest X-ray image. First, provide your step-by-step reasoning "
        "for the diagnosis within <thinking> tags. Then, provide the final classification "
        "labels within <answer> tags. The classification should be a comma-separated list "
        f"of applicable labels from the following: {', '.join(LABEL_COLS)}. " # Usa LABEL_COLS
        "Example format: <thinking>The heart appears enlarged...</thinking><answer>Cardiomegaly, Pleural Effusion</answer>"
        "IMPORTANT: Your entire response must contain ONLY the <thinking>...</thinking><answer>...</answer> structure. "
        "Do not include any text before the <thinking> tag or after the </answer> tag."
    )
    
    # Usa una lambda per passare tokenizer e instruction_text a prepare_prompt
    # Questo è necessario perché .map non passa argomenti extra direttamente in modo pulito
    # se non tramite functools.partial o una lambda.
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
        per_device_eval_batch_size=args.per_device_train_batch_size, # Mantenuto coerente
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        sync_ref_model=True,
        logging_steps=1,
        max_completion_length=2048,
        temperature=1.1,
        # evaluation_strategy="steps", # Abilitare se si desidera la valutazione durante l'addestramento
        # eval_steps=50,
        # save_strategy="steps",
        # save_steps=50,
        report_to=args.report_to if not args.disable_wandb else "none",
        output_dir="outputs", # Potrebbe essere reso configurabile anche questo
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn, # reward_fn ora usa LABEL_COLS globalmente
        args=config,
        train_dataset=proc_ds_train,
        eval_dataset=proc_ds_eval,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # ---------------------------------------------------------------------
    # 5.  SAVE LoRA ADAPTER  ──────────────────────────────────────────────
    # ---------------------------------------------------------------------
    print(f"Saving LoRA adapter to ./{args.output_lora_path}")
    model.save_lora(args.output_lora_path)
    print(f"LoRA adapter saved to ./{args.output_lora_path}")

    if args.report_to == "wandb" and not args.disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
