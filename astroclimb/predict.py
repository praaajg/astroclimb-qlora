from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from .data import prepare_test_frame
from .labels import LABELS, parse_label
from .model import attach_adapter, load_base_model, load_processor
from .prompts import build_inference_messages, build_train_messages, collect_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroCLIMB test submission")
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument(
        "--adapter",
        default="ShivRamSaud/gemma4-e4b-qlora-astroclimb-v2-fresh",
    )
    parser.add_argument("--adapter-subfolder", default="adapter_best")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", choices=["inference", "train"], default="inference")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def generate_label(
    model, processor, row, max_new_tokens: int, prompt: str = "inference"
) -> str:
    builder = build_train_messages if prompt == "train" else build_inference_messages
    messages = builder(row)
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
    return parse_label(decoded)


def main() -> None:
    args = parse_args()
    frame = prepare_test_frame(pd.read_csv(args.test_csv))
    if "id" not in frame.columns:
        frame["id"] = range(len(frame))
    if args.limit > 0:
        frame = frame.head(args.limit).reset_index(drop=True)
    processor = load_processor(args.model_id)
    base = load_base_model(args.model_id)
    model = attach_adapter(base, args.adapter, args.adapter_subfolder)
    model.eval()
    predictions = []
    for index in range(len(frame)):
        row = frame.iloc[index]
        try:
            prediction = generate_label(
                model, processor, row, args.max_new_tokens, args.prompt
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            gc.collect()
            try:
                prediction = generate_label(
                    model, processor, row, args.max_new_tokens, args.prompt
                )
            except RuntimeError:
                prediction = "unrelated_papers"
        predictions.append(prediction)
        if index % 50 == 0:
            torch.cuda.empty_cache()
        if (index + 1) % 200 == 0:
            print(f"predicted {index + 1}/{len(frame)}")
    submission = pd.DataFrame({"id": frame["id"].values})
    for label in LABELS:
        submission[label] = [int(prediction == label) for prediction in predictions]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"wrote {output} rows={len(submission)}")


if __name__ == "__main__":
    main()
