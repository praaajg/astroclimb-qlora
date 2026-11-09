"""Reference Hugging Face Trainer pipeline for AstroCLIMB QLoRA fine-tuning.

This is a clean-room implementation of the protocol described in the paper. The
paper's original training script was not released; see the implementation note
in the README. GPU dependencies are imported lazily so the rest of the package
remains testable on CPU-only machines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data import (
    detect_object_kind,
    load_pairs,
    prepare_image,
    truncate_caption,
)
from .model import (
    DEFAULT_MODEL_ID,
    DEFAULT_TARGET_MODULES,
    build_lora_config,
    load_base_model,
    load_processor,
)
from .prompt import FigureObject, build_messages
from .split import stratified_validation_split


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return config


def _make_object(name: str, value: Any, kind: str) -> FigureObject:
    if kind == "figure":
        return FigureObject(name=name, kind="figure", image=prepare_image(value))
    return FigureObject(
        name=name,
        kind="caption",
        text=truncate_caption(str(value)),
    )


def row_to_messages(row: pd.Series, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one competition row to the shared chat message list."""

    data = config["data"]
    a_col = data["object_a_col"]
    b_col = data["object_b_col"]
    kind_a = row.get(data["type_a_col"]) or detect_object_kind(row[a_col])
    kind_b = row.get(data["type_b_col"]) or detect_object_kind(row[b_col])

    object_a = _make_object("A", row[a_col], str(kind_a).lower())
    object_b = _make_object("B", row[b_col], str(kind_b).lower())
    return build_messages(object_a, object_b)


class PairDataset:
    """Torch dataset yielding processor-encoded chat examples."""

    def __init__(
        self,
        frame: pd.DataFrame,
        processor: Any,
        config: dict[str, Any],
        *,
        with_labels: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor
        self.config = config
        self.with_labels = with_labels
        self.mask_prompt = bool(config["training"].get("mask_prompt_tokens", False))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.frame.iloc[index]
        messages = row_to_messages(row, self.config)
        label = str(row[self.config["data"]["label_col"]])

        full_messages = list(messages)
        if self.with_labels:
            full_messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": label}]}
            )

        encoded = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=not self.with_labels,
            return_dict=True,
            return_tensors="pt",
        )

        item: dict[str, Any] = {}
        for key, value in encoded.items():
            if (
                isinstance(value, torch.Tensor)
                and key in {"input_ids", "attention_mask", "token_type_ids"}
                and value.ndim > 1
                and value.shape[0] == 1
            ):
                item[key] = value.squeeze(0)
            else:
                item[key] = value

        if self.with_labels:
            labels = item["input_ids"].clone()
            if self.mask_prompt:
                prompt_encoded = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                prompt_length = int(prompt_encoded["input_ids"].shape[-1])
                labels[:prompt_length] = -100
            item["labels"] = labels
        return item


def make_collator(tokenizer: Any):
    """Pad text tensors and concatenate image tensors for the VLM."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0

        max_length = max(int(item["input_ids"].shape[-1]) for item in batch)
        input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
        labels = torch.full((len(batch), max_length), -100, dtype=torch.long)

        for row_index, item in enumerate(batch):
            length = int(item["input_ids"].shape[-1])
            input_ids[row_index, :length] = item["input_ids"]
            attention_mask[row_index, :length] = item.get(
                "attention_mask", torch.ones(length, dtype=torch.long)
            )
            if "labels" in item:
                labels[row_index, :length] = item["labels"]

        output: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "labels" in batch[0]:
            output["labels"] = labels
        if "pixel_values" in batch[0]:
            output["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        if "image_sizes" in batch[0]:
            output["image_sizes"] = torch.cat([item["image_sizes"] for item in batch], dim=0)
        return output

    return collate


def build_training_arguments(config: dict[str, Any], tokenizer: Any):
    from transformers import TrainingArguments

    training = config["training"]
    kwargs: dict[str, Any] = {
        "output_dir": training["output_dir"],
        "max_steps": int(training["max_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "lr_scheduler_type": training.get("scheduler", "cosine"),
        "warmup_steps": int(training.get("warmup_steps", 0)),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "eval_strategy": "steps",
        "eval_steps": int(training["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(training["save_steps"]),
        "logging_steps": int(training.get("logging_steps", 20)),
        "bf16": bool(training.get("bf16", True)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": int(training.get("seed", 42)),
        "data_seed": int(training.get("seed", 42)),
        "prediction_loss_only": True,
    }
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        # Transformers renamed evaluation_strategy -> eval_strategy in 4.46.
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the AstroCLIMB QLoRA adapter")
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument("--train-csv", help="Override data.train_csv")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    args = parser.parse_args(argv)

    config = read_config(args.config)
    if args.train_csv:
        config["data"]["train_csv"] = args.train_csv
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, Trainer, set_seed

    training = config["training"]
    lora = config["lora"]
    model_config = config["model"]
    set_seed(int(training.get("seed", 42)))

    frame = load_pairs(
        config["data"]["train_csv"],
        id_col=config["data"]["id_col"],
        object_a_col=config["data"]["object_a_col"],
        object_b_col=config["data"]["object_b_col"],
        label_col=config["data"]["label_col"],
        type_a_col=config["data"]["type_a_col"],
        type_b_col=config["data"]["type_b_col"],
    )
    train_frame, validation_frame = stratified_validation_split(
        frame,
        label_col=config["data"]["label_col"],
        n_per_class=int(config["data"]["val_per_class"]),
        seed=int(config["data"]["seed"]),
    )
    print(f"Train rows: {len(train_frame):,} | Validation rows: {len(validation_frame):,}")

    processor = load_processor(model_config.get("id", DEFAULT_MODEL_ID))
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = load_base_model(
        model_config.get("id", DEFAULT_MODEL_ID),
        load_in_4bit=bool(model_config.get("load_in_4bit", True)),
        quant_type=model_config.get("quant_type", "nf4"),
        double_quant=bool(model_config.get("double_quant", True)),
        compute_dtype=model_config.get("compute_dtype", "bfloat16"),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
    )
    model = get_peft_model(
        model,
        build_lora_config(
            r=int(lora["r"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora["dropout"]),
            target_modules=lora.get("target_modules", DEFAULT_TARGET_MODULES),
            bias=lora.get("bias", "none"),
            task_type=lora.get("task_type", "CAUSAL_LM"),
        ),
    )
    model.print_trainable_parameters()

    train_dataset = PairDataset(train_frame, processor, config, with_labels=True)
    validation_dataset = PairDataset(validation_frame, processor, config, with_labels=True)
    training_args = build_training_arguments(config, processor.tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=make_collator(processor.tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(training.get("early_stopping_patience", 3)),
                early_stopping_threshold=float(training.get("early_stopping_threshold", 0.001)),
            )
        ],
    )
    trainer.train()

    output_dir = Path(training["output_dir"])
    adapter_dir = output_dir / "adapter_best"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Saved adapter to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
