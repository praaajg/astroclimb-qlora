from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .data import (
    DEFAULT_DATASET,
    load_frame,
    prepare_train_frame,
    sample_balanced,
)
from .labels import LABELS, parse_label
from .model import attach_adapter, load_base_model, load_processor
from .prompts import build_inference_messages, collect_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroCLIMB balanced validation")
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument(
        "--adapter",
        default="ShivRamSaud/gemma4-e4b-qlora-astroclimb-v2-fresh",
    )
    parser.add_argument("--adapter-subfolder", default="adapter_best")
    parser.add_argument("--train-csv", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="outputs/astroclimb_eval")
    parser.add_argument("--val-per-class", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--time-budget-h", type=float, default=6.5)
    return parser.parse_args()


def generate_label(model, processor, row, max_new_tokens: int) -> tuple[str, str]:
    messages = build_inference_messages(row)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = collect_images(messages)
    if images:
        inputs = processor(
            text=[text], images=[images], padding=True, return_tensors="pt"
        )
    else:
        inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    decoded = processor.batch_decode(
        output[:, length:], skip_special_tokens=True
    )[0]
    return parse_label(decoded), decoded


def main() -> None:
    args = parse_args()
    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = load_processor(args.model_id)
    base = load_base_model(args.model_id)
    model = attach_adapter(base, args.adapter, args.adapter_subfolder)
    model.eval()
    frame = prepare_train_frame(load_frame(args.train_csv))
    val = sample_balanced(
        frame, n_per_class=args.val_per_class, seed=args.seed
    ).reset_index(drop=True)
    predictions = []
    raw_rows = []
    n_oom = 0
    n_parse = 0
    for index in range(len(val)):
        if time.time() - start_time > args.time_budget_h * 3600:
            print(f"time budget reached at {index}/{len(val)}")
            break
        row = val.iloc[index]
        try:
            prediction, decoded = generate_label(
                model, processor, row, args.max_new_tokens
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            gc.collect()
            try:
                prediction, decoded = generate_label(
                    model, processor, row, args.max_new_tokens
                )
            except RuntimeError:
                prediction, decoded = "unrelated_papers", ""
                n_oom += 1
        if prediction == "unrelated_papers" and "unrelated_papers" not in decoded.lower():
            n_parse += 1
        predictions.append(prediction)
        raw_rows.append(
            {
                "idx": index,
                "id": int(row["id"]) if "id" in row else index,
                "true": row["label"],
                "pred": prediction,
                "raw": decoded,
            }
        )
        if index % 50 == 0:
            torch.cuda.empty_cache()
        if (index + 1) % 200 == 0:
            elapsed = (time.time() - start_time) / 3600
            print(f"scored {index + 1}/{len(val)} elapsed={elapsed:.2f}h")
    y_true = val["label"].values[: len(predictions)]
    y_pred = np.array(predictions)
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=LABELS, zero_division=0
    )
    per_class_precision = precision_score(
        y_true, y_pred, average=None, labels=LABELS, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=LABELS, zero_division=0
    )
    per_pair_type = {}
    for pair_type in ["TXT-IMG", "IMG-IMG", "TXT-TXT"]:
        mask = val["pair_type"].values[: len(predictions)] == pair_type
        if mask.sum() == 0:
            continue
        per_pair_type[pair_type] = {
            "rows": int(mask.sum()),
            "macro_f1": float(
                f1_score(
                    y_true[mask], y_pred[mask], average="macro", zero_division=0
                )
            ),
        }
    metrics = {
        "model": args.model_id,
        "adapter": f"{args.adapter}/{args.adapter_subfolder}",
        "split": {
            "rows": len(y_true),
            "per_class": args.val_per_class,
            "seed": args.seed,
        },
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(per_class_precision[position]),
                "recall": float(per_class_recall[position]),
                "f1": float(per_class_f1[position]),
            }
            for position, label in enumerate(LABELS)
        },
        "per_pair_type": per_pair_type,
        "oom": n_oom,
        "parse_fallback": n_parse,
        "time_h": float((time.time() - start_time) / 3600),
    }
    np.save(output_dir / "preds_val.npy", y_pred)
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in raw_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
