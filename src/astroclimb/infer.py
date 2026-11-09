"""Run AstroCLIMB inference and write one-hot submission predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .data import (
    detect_object_kind,
    load_pairs,
    prepare_image,
    truncate_caption,
)
from .labels import LABELS, parse_label
from .model import attach_adapter, load_base_model, load_processor
from .prompt import FigureObject, build_messages
from .train import read_config


def _make_object(name: str, value: Any, kind: str) -> FigureObject:
    if kind == "figure":
        return FigureObject(name=name, kind="figure", image=prepare_image(value))
    return FigureObject(name=name, kind="caption", text=truncate_caption(str(value)))


def messages_for_row(row: pd.Series, config: dict[str, Any]) -> list[dict[str, Any]]:
    data = config["data"]
    value_a = row[data["object_a_col"]]
    value_b = row[data["object_b_col"]]
    kind_a = row.get(data["type_a_col"]) or detect_object_kind(value_a)
    kind_b = row.get(data["type_b_col"]) or detect_object_kind(value_b)
    object_a = _make_object("A", value_a, str(kind_a).lower())
    object_b = _make_object("B", value_b, str(kind_b).lower())
    return build_messages(object_a, object_b)


def generate_label(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 32,
) -> str:
    """Generate one label with greedy decoding and regex parsing."""

    import torch

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_length = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = processor.batch_decode(
        output[:, prompt_length:],
        skip_special_tokens=True,
    )[0]
    return parse_label(generated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AstroCLIMB inference")
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--input", help="Override data.test_csv")
    parser.add_argument("--adapter", help="Local path or Hub id of the LoRA adapter")
    parser.add_argument("--output", default="outputs/test_predictions.csv")
    parser.add_argument("--limit", type=int, help="Only score the first N rows (smoke test)")
    args = parser.parse_args(argv)

    config = read_config(args.config)
    input_path = args.input or config["data"]["test_csv"]
    frame = load_pairs(
        input_path,
        id_col=config["data"]["id_col"],
        object_a_col=config["data"]["object_a_col"],
        object_b_col=config["data"]["object_b_col"],
        label_col=config["data"]["label_col"],
        type_a_col=config["data"]["type_a_col"],
        type_b_col=config["data"]["type_b_col"],
    )
    if args.limit:
        frame = frame.head(args.limit)

    processor = load_processor(config["model"].get("id"))
    model = load_base_model(
        config["model"].get("id"),
        load_in_4bit=bool(config["model"].get("load_in_4bit", True)),
        quant_type=config["model"].get("quant_type", "nf4"),
        double_quant=bool(config["model"].get("double_quant", True)),
        compute_dtype=config["model"].get("compute_dtype", "bfloat16"),
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
    )
    if args.adapter:
        model = attach_adapter(model, args.adapter)
    model.eval()

    id_col = config["data"]["id_col"]
    max_new_tokens = int(config["inference"].get("max_new_tokens", 32))
    predictions: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        label = generate_label(
            model,
            processor,
            messages_for_row(row, config),
            max_new_tokens=max_new_tokens,
        )
        predictions.append(
            {
                id_col: row[id_col],
                **{class_name: int(label == class_name) for class_name in LABELS},
            }
        )
        if (index + 1) % 100 == 0:
            print(f"Scored {index + 1:,}/{len(frame):,} pairs")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output, index=False)
    print(f"Wrote {len(predictions):,} predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
