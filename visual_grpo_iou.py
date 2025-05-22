# -*- coding: utf-8 -*-
"""
GRPO fine‑tuning of Qwen2.5‑VL‑3B on VinDr‑CXR multi‑label + localization
-----------------------------------------------------------------------
This script extends the previous CheXpert‑only pipeline with:
  • loading of the VinDr‑CXR dataset (bounding boxes + global labels)
  • extraction of ground‑truth bounding boxes
  • updated prompt that asks the model to output **both** diagnostic labels and
    one bounding box per abnormality, inside a constrained XML‑like format
  • new IoU‑based reward component that compares the predicted and reference
    bounding boxes and returns a score in [‑1, 1]
  • updated composite reward that blends (format, classification, IoU)
  • all previous CLI flags preserved + ``--iou_weight`` to control balance
  • LoRA export identical to the original script

Run in env:
    pip install unsloth "scikit-learn>=1.4" datasets pillow
"""
from __future__ import annotations
import re, os, json, argparse, math, torch
from collections import defaultdict
from typing import List, Dict, Tuple
from datasets import load_dataset
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# 0.  GLOBAL HYPERPARAMETERS & CONFIGURATIONS  ─────────────────────────
# ---------------------------------------------------------------------
DEFAULT_MODEL_NAME = "unsloth/gemma-3-4b-it"
DEFAULT_N_SAMPLES_TRAIN = 3
DEFAULT_MAX_STEPS = 500
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_OUTPUT_LORA_PATH = "vindr_grpo_lora"
DEFAULT_REPORT_TO = "wandb"
DEFAULT_IOU_WEIGHT = 0.3  # weight of IoU in the composite reward

# ---------------------------------------------------------------------
# 1.  DATASET CONSTANTS  ───────────────────────────────────────────────
# ---------------------------------------------------------------------
# VinDr‑CXR local‑label ontology (22 findings). Refer to the public paper.
LABEL_COLS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Clavicle fracture", "Consolidation", "ILD", "Infiltration",
    "Lung Opacity", "Lung cavity", "Lung cyst", "Lung lesion",
    "Mediastinal shift", "Nodule/Mass", "Pleural effusion",
    "Pleural thickening", "Pneumoperitoneum", "Pneumothorax",
    "Pulmonary fibrosis", "Rib fracture", "Other lesion",
    "No finding"  # keep for completeness
]

# compile once
LABEL_REGEX = re.compile(r"|".join([re.escape(c) for c in LABEL_COLS]), re.I)
ANSWER_REGEX = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
BBOX_REGEX = re.compile(r"([\w/\s]+?):\s*\[\s*(\d+),(\d+),(\d+),(\d+)\s*]", re.I)
STRUCTURE_REGEX = re.compile(
    r"^\s*<thinking>(?:(?!</?(?:thinking|answer)>).)*</thinking>\s*<answer>(?:(?!</?(?:thinking|answer)>).)*</answer>\s*$",
    re.I | re.S,
)

# ---------------------------------------------------------------------
# 2.  ARGPARSE  ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Fine‑tune VLM with GRPO (bbox+class)")
    p.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--n_samples_train", type=int, default=DEFAULT_N_SAMPLES_TRAIN)
    p.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument("--num_generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    p.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE)
    p.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--output_lora_path", default=DEFAULT_OUTPUT_LORA_PATH)
    p.add_argument("--report_to", default=DEFAULT_REPORT_TO)
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--iou_weight", type=float, default=DEFAULT_IOU_WEIGHT, help="Weight of IoU reward component (0‑1)")
    return p.parse_args()

# ---------------------------------------------------------------------
# 3.  DATA HELPERS  ────────────────────────────────────────────────────
# ---------------------------------------------------------------------

def prepare_prompt(sample, tokenizer, instruction_text):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_text},
                {"type": "image", "image": sample["image"]},
            ],
        }
    ]
    sample["prompt"] = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return sample


def extract_gt(sample):
    # image‑level present labels (dedup)
    gt_labels = set()
    gt_bboxes: Dict[str, List[Tuple[int,int,int,int]]] = defaultdict(list)

    for ann in sample["annotations"]:  # VinDr‑CXR field
        label = ann["label"]
        if label not in LABEL_COLS:
            continue
        gt_labels.add(label)
        # ann has [x_min, y_min, x_max, y_max]
        gt_bboxes[label].append(tuple(ann["bbox"]))

    sample["gt_labels"] = list(gt_labels)
    sample["gt_bboxes"] = {k: v for k, v in gt_bboxes.items()}
    return sample

# ---------------------------------------------------------------------
# 4.  PREDICTION PARSERS  ─────────────────────────────────────────────
# ---------------------------------------------------------------------

def parse_pred_labels(text: str) -> List[str]:
    ans_match = ANSWER_REGEX.search(text)
    if not ans_match:
        return []
    answer_content = ans_match.group(1)
    return list({m.group(0).title() for m in LABEL_REGEX.finditer(answer_content)})


def parse_pred_bboxes(text: str) -> Dict[str, List[Tuple[int,int,int,int]]]:
    ans_match = ANSWER_REGEX.search(text)
    if not ans_match:
        return {}
    answer_content = ans_match.group(1)
    boxes: Dict[str, List[Tuple[int,int,int,int]]] = defaultdict(list)
    for lbl, x1, y1, x2, y2 in BBOX_REGEX.findall(answer_content):
        label_norm = lbl.title().strip()
        if label_norm in LABEL_COLS:
            boxes[label_norm].append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

# ---------------------------------------------------------------------
# 5.  REWARD COMPONENTS  ──────────────────────────────────────────────
# ---------------------------------------------------------------------

def format_reward(text: str) -> float:
    ok = STRUCTURE_REGEX.match(text) and ANSWER_REGEX.search(text)
    return 1.0 if ok else -1.0


def classification_reward(pred_labels: List[str], gt_labels: List[str]) -> float:
    y_true = [1 if l in gt_labels else 0 for l in LABEL_COLS]
    y_pred = [1 if l in pred_labels else 0 for l in LABEL_COLS]
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 0.5
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return 2 * f1 - 1  # scale


def compute_iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union else 0.0


def iou_reward(pred_boxes: Dict[str, List[Tuple[int,int,int,int]]],
               gt_boxes: Dict[str, List[Tuple[int,int,int,int]]]) -> float:
    # compute mean max IoU over all GT boxes
    if not gt_boxes:
        return 0.5 if not pred_boxes else -1.0
    ious = []
    for lbl, gt_list in gt_boxes.items():
        pred_list = pred_boxes.get(lbl, [])
        for g in gt_list:
            best = max((compute_iou(g, p) for p in pred_list), default=0.0)
            ious.append(best)
    if not ious:
        return -1.0
    mean_iou = sum(ious)/len(ious)
    return 2 * mean_iou - 1  # scale to [-1,1]

# ---------------------------------------------------------------------
# 6.  COMPOSITE REWARD  ───────────────────────────────────────────────
# ---------------------------------------------------------------------

def build_reward_fn(iou_weight: float):
    fmt_w = 0.4
    cls_w = 0.6 - iou_weight  # remaining weight

    def _fn(prompts: List[str], completions: List[str], **kwargs):
        gt_labels_list = kwargs.get("gt_labels")
        gt_bboxes_list = kwargs.get("gt_bboxes")
        rewards = []
        for text, gt_labels, gt_bboxes in zip(completions, gt_labels_list, gt_bboxes_list):
            fmt_r = format_reward(text)
            pred_labels = parse_pred_labels(text)
            cls_r = classification_reward(pred_labels, gt_labels)
            pred_boxes = parse_pred_bboxes(text)
            iou_r = iou_reward(pred_boxes, gt_bboxes)
            combined = fmt_w * fmt_r + cls_w * cls_r + iou_weight * iou_r
            # small bonus if both labels & boxes set‑exact match
            if set(pred_labels) == set(gt_labels) and iou_r > 0.8:
                combined += 0.5
            rewards.append(combined)
        return rewards
    return _fn

# ---------------------------------------------------------------------
# 7.  MAIN  ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    from unsloth import FastVisionModel
    from trl import GRPOConfig, GRPOTrainer
    import wandb

    if args.report_to == "wandb" and not args.disable_wandb:
        wandb.init(project="MedicalVLM‑VinDr‑BBox", config=vars(args))
    else:
        os.environ["WANDB_DISABLED"] = "true"

    print("Loading VinDr‑CXR dataset…")
    raw_train = load_dataset("vinbigdata/vindr-cxr", split="train")
    raw_eval = load_dataset("vinbigdata/vindr-cxr", split="test")

    if args.n_samples_train > 0:
        raw_train = raw_train.shuffle(seed=42).select(range(args.n_samples_train))

    print("Loading model…")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        random_state=3407,
    )

    # instruction
    instruction_text = (
        "Analyze the chest X‑ray. Provide reasoning inside <thinking> tags, then "
        "inside <answer> give a comma‑separated list of findings with bounding "
        "boxes in the form Label: [x1,y1,x2,y2]; e.g. "
        "<answer>Cardiomegaly: [50,80,450,520]; Pleural effusion: [60,600,500,900]</answer>. "
        "No extra text before <thinking> or after </answer>."
    )

    prep_fn = lambda s: prepare_prompt(s, tokenizer, instruction_text)

    proc_train = (raw_train.map(extract_gt)
                          .map(prep_fn))
    proc_eval  = (raw_eval.map(extract_gt)
                          .map(prep_fn))

    config = GRPOConfig(
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=args.report_to if not args.disable_wandb else "none",
        output_dir="outputs_vindr",
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=build_reward_fn(args.iou_weight),
        args=config,
        train_dataset=proc_train,
        eval_dataset=proc_eval,
        processing_class=tokenizer,
    )

    print("Start training…")
    trainer.train()
    print("Finished training, saving LoRA…")
    model.save_lora(args.output_lora_path)

    if args.report_to == "wandb" and not args.disable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
