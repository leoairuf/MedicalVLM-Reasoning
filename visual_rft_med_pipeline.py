#!/usr/bin/env python
"""Visual‑RFT end‑to‑end pipeline (medical domain)
================================================
CLI usage
---------
# ❶ scarica dati + (opz.) bbox pseudo‑auto‑generate
python visual_rft_med_pipeline.py prepare   
    --datasets chexpert vindr_cxr   
    --pseudo_bbox chexpert          # usa MedSAM2 per creare bbox su CheXpert

# ❷ fine‑tuning con GRPO+ (LoRA‑4bit, vLLM)
python visual_rft_med_pipeline.py train_grpo   
    --steps 25000 --generations 4

# ❸ ablation SFT puro
python visual_rft_med_pipeline.py train_sft   
    --epochs 2

# ❹ valutazione AUROC / mAP vs competitor
python visual_rft_med_pipeline.py evaluate    
    --ckpt runs/grpo_final --baseline_chexnet path/to/chexnet.ckpt \
    --baseline_retinanet path/to/retinanet.ckpt

Dipendenze
~~~~~~~~~~
```
pip install "unsloth>=2025.4" "vllm>=0.4" "pytorch-lightning"
            "transformers>=4.48" "datasets" "segment-anything==2.*"
            "medsam-segment-anything" "evaluate" "scikit-learn"
```
"""

import argparse, json, os, random, math, re, shutil, warnings, pathlib
from functools import partial
from typing import List, Dict, Tuple

import torch, numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image
from tqdm import tqdm

# ------------------------------ segmenter ----------------------------------
try:
    from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator
    _SEG_OK = True
except ImportError:
    _SEG_OK = False

# ------------------------------ utils --------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ------------------------------ CONFIG ------------------------------------
IMAGE_SIZE   = 448
MODEL_ID     = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_R       = 64
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 2048

_CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
]

# ------------------------------ PREPARE ------------------------------------

def _resize(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def _pseudo_bbox_mask(img: Image.Image, sam: SamAutomaticMaskGenerator):
    masks = sam.generate(np.asarray(img))
    if len(masks) == 0:
        return None
    best = max(masks, key=lambda m: m['area'])
    x, y, w, h = best['bbox']  # xywh in original resolution
    scale_x = IMAGE_SIZE / img.width
    scale_y = IMAGE_SIZE / img.height
    return [x*scale_x, y*scale_y, w*scale_x, h*scale_y]


def prepare_chexpert(split: str, pseudo_bbox: bool=False, sam=None):
    ds = load_dataset("danjacobellis/chexpert", split=split)

    def _map(ex):
        img = _resize(Image.open(ex['path']))
        ex['pixel_values'] = np.asarray(img)
        ex['labels'] = [int(float(ex[l]==1.0)) for l in _CHEXPERT_LABELS]
        ex['task'] = 'cls'
        if pseudo_bbox and sam is not None:
            bbox = _pseudo_bbox_mask(img, sam)
            if bbox is not None:
                ex['gt_bbox'] = bbox; ex['task'] = 'det'  # enables R_det too
        return ex

    return ds.map(_map, num_proc=8)


def prepare_vindr(split: str):
    ds = load_dataset("lufficc/vindr-cxr-bbox", split=split)  # contains bbox + label

    def _map(ex):
        img = _resize(Image.open(ex['image_path']))
        ex['pixel_values'] = np.asarray(img)
        # VinDr: one bbox per finding; keep first for simplicity
        ex['gt_bbox'] = [float(ex['x']), float(ex['y']), float(ex['w']), float(ex['h'])]
        ex['task'] = 'det'
        # unify to single class label (finding presence)
        ex['labels'] = [1]  # placeholder so dataset columns align
        return ex

    return ds.map(_map, num_proc=8)


def build_prompts(batch):
    out_prompt = []
    answers    = []
    for task, bbox, labels in zip(batch['task'], batch['gt_bbox'], batch['labels']):
        if task == 'cls':
            p = (
                "<image>\nClassify the chest X‑ray for the findings in order: "
                f"{', '.join(_CHEXPERT_LABELS)}. "
                "Return ONLY JSON in <think>/<answer> tags as {'labels':[..]}"
            )
            ans = json.dumps({"labels": labels})
        else:
            p = (
                "<image>\nLocate any abnormal opacity. Return SINGLE JSON bbox:" \
                " <think></think><answer>{'x':..., 'y':..., 'width':..., 'height':...}</answer>"
            )
            ans = json.dumps({"bbox": bbox})
        out_prompt.append([{"role":"user","content":p}])
        answers.append(ans)
    batch['prompt'] = out_prompt
    batch['answer'] = answers
    return batch


def command_prepare(args):
    os.makedirs("data_cache", exist_ok=True)
    sam = None
    if args.pseudo_bbox and _SEG_OK:
        sam_path = sam_model_registry("vit_b", checkpoint="medsam_vit_b.pth")
        sam = SamAutomaticMaskGenerator(sam_path)
    print("→ Preparo CheXpert …")
    chex = prepare_chexpert("train", args.pseudo_bbox, sam)
    print("→ Preparo VinDr‑CXR …")
    vindr = prepare_vindr("train")
    dataset = concatenate_datasets([chex, vindr]).map(build_prompts, batched=True, batch_size=256)
    dataset.save_to_disk("data_cache/train_mix")
    print("✅ Dataset salvato in data_cache/train_mix (", len(dataset), "campioni )")

# ---------------------------- REWARD FUNCS ---------------------------------

def iou(boxA: List[float], boxB: List[float]):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]); yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0.
    return inter / (boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter)


def reward_fn(sample, generation):
    # quick json safe‑load
    def _safe(js):
        try: return json.loads(js)
        except Exception: return None

    fmt_ok = bool(re.search(r"<think>[\s\S]+</think>\s*<answer>[\s\S]+</answer>", generation))
    if sample['task'] == 'cls':
        obj = _safe(generation[generation.find('<answer>')+8:generation.rfind('</answer>')])
        if obj is None or 'labels' not in obj:
            return -1.0
        acc = sum(int(a==b) for a,b in zip(obj['labels'], sample['labels'])) / len(_CHEXPERT_LABELS)
        return acc + (0.1 if fmt_ok else -0.5)
    else:
        obj = _safe(generation[generation.find('<answer>')+8:generation.rfind('</answer>')])
        if obj is None: return -1.0
        try:
            bbox = [float(obj[k]) for k in ('x','y','width','height')]
            i = iou(bbox, sample['gt_bbox'])
            conf = float(obj.get('conf',0.5))
            R_conf = conf if i>0 else 1-conf
            return i + R_conf + (0.1 if fmt_ok else -0.5)
        except Exception:
            return -1.0

# ---------------------------- TRAINING -------------------------------------

def _load_model(lora=True):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True,
        use_flash_attention_2=True,
        fast_inference=True,
        trust_remote_code=True,
    )
    if lora:
        model = FastLanguageModel.get_peft_model(model, r=LORA_R, lora_alpha=LORA_ALPHA,
                                                  lora_dropout=LORA_DROPOUT, bias="none")
    return model, tokenizer


def command_train_grpo(args):
    from unsloth import FastRLHFTrainer, GRPOConfig
    ds = Dataset.load_from_disk("data_cache/train_mix")
    model, tokenizer = _load_model()
    cfg = GRPOConfig(use_vllm=True, num_generations=args.generations, learning_rate=5e-6,
                     max_steps=args.steps)
    trainer = FastRLHFTrainer(model=model, tokenizer=tokenizer, train_dataset=ds,
                              reward_function=reward_fn, batch_size=1, grpo_config=cfg,
                              image_column="pixel_values")
    model.gradient_checkpointing_enable(); trainer.fit()
    os.makedirs("runs/grpo", exist_ok=True)
    model.save_lora("runs/grpo/lora"); model.save_pretrained_merged("runs/grpo/full16", tokenizer)
    print("✅ GRPO fine‑tuning finito — artefatti in runs/grpo")

# ----------------------------- SFT -----------------------------------------

def command_train_sft(args):
    from unsloth import FastLanguageModel
    ds = Dataset.load_from_disk("data_cache/train_mix")
    model, tokenizer = _load_model(lora=True)
    def _collate(batch):
        return tokenizer([ex['prompt'] for ex in batch], return_tensors='pt', padding=True, truncation=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train();
    for epoch in range(args.epochs):
        for step, ex in enumerate(tqdm(loader, desc=f"SFT epoch{epoch+1}")):
            out = model(**{k:v.cuda() for k,v in ex.items()}, labels=ex['input_ids'].cuda())
            out.loss.backward(); opt.step(); opt.zero_grad()
    os.makedirs("runs/sft", exist_ok=True)
    model.save_lora("runs/sft/lora")
    print("✅ SFT finito — artefatti in runs/sft")

# ----------------------------- EVAL ----------------------------------------

def eval_cls(model, tokenizer, split="validation"):
    chex = load_dataset("danjacobellis/chexpert", split=split)
    # simple AUROC on 5 labels
    from sklearn.metrics import roc_auc_score
    scores = [[] for _ in _CHEXPERT_LABELS]; gts=[[] for _ in _CHEXPERT_LABELS]
    for ex in tqdm(chex.select(range(1000)), desc="eval cls"):
        prompt = [{"role":"user","content":"<image>\nSame instruction as training."}]
        pix = torch.tensor(_resize(Image.open(ex['path'])))
        tokens = tokenizer(prompt, return_tensors='pt').to('cuda')
        gen = model.generate(**tokens, images=pix.unsqueeze(0).to('cuda'), max_new_tokens=64)
        txt = tokenizer.decode(gen[0], skip_special_tokens=True)
        try:
            pred = json.loads(txt[txt.find('{'):txt.find('}')+1])["labels"]
        except Exception:
            pred=[0]*5
        for i,l in enumerate(_CHEXPERT_LABELS):
            scores[i].append(pred[i]); gts[i].append(int(float(ex[l]==1.0)))
    auc = [roc_auc_score(gts[i], scores[i]) for i in range(5)]
    return float(np.mean(auc))


def eval_det(model, tokenizer, split="test"):
    vindr = prepare_vindr(split)
    TP, FP, FN = 0,0,0
    for ex in tqdm(vindr, desc="eval det"):
        prompt = [{"role":"user","content":"<image>\nLocate any abnormal opacity. Give bbox JSON."}]
        pix = torch.tensor(ex['pixel_values']).unsqueeze(0).to('cuda')
        tokens = tokenizer(prompt, return_tensors='pt').to('cuda')
        txt = tokenizer.decode(model.generate(**tokens, images=pix, max_new_tokens=64)[0], skip_special_tokens=True)
        try:
            pred = json.loads(txt[txt.find('{'):txt.find('}')+1])
            i = iou([pred[k] for k in ('x','y','width','height')], ex['gt_bbox'])
            if i>0.5: TP+=1
            else: FP+=1; FN+=1
        except Exception:
            FN+=1
    precision = TP/(TP+FP+1e-9); recall=TP/(TP+FN+1e-9)
    return {'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall+1e-9)}


def command_evaluate(args):
    from unsloth import FastLanguageModel
    model, tok = _load_model(lora=False)  # load base
    model.load_lora(args.ckpt)
    cls_auc = eval_cls(model, tok)
    det_metrics = eval_det(model, tok)
    print("— OUR GRPO model — AUROC avg:", cls_auc, " Detection F1:", det_metrics['f1'])
    # baseline CheXNet
    if args.baseline_chexnet:
        print("(placeholder) Carica CheXNet e calcola AUROC …")
    # baseline RetinaNet
    if args.baseline_retinanet:
        print("(placeholder) Carica RetinaNet e calcola F1 …")

# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()

    sp = sub.add_parser('prepare');
    sp.add_argument('--datasets', nargs='+', default=['chexpert','vindr_cxr'])
    sp.add_argument('--pseudo_bbox', action='store_true')
    sp.set_defaults(func=command_prepare)

    sp = sub.add_parser('train_grpo')
    sp.add_argument('--steps', type=int, default=25000)
    sp.add_argument('--generations', type=int, default=4)
    sp.set_defaults(func=command_train_grpo)

    sp = sub.add_parser('train_sft')
    sp.add_argument('--epochs', type=int, default=2)
    sp.set_defaults(func=command_train_sft)

    sp = sub.add_parser('evaluate')
    sp.add_argument('--ckpt', required=True)
    sp.add_argument('--baseline_chexnet')
    sp.add_argument('--baseline_retinanet')
    sp.set_defaults(func=command_evaluate)

    args = p.parse_args(); args.func(args)

if __name__ == "__main__":
    main()
