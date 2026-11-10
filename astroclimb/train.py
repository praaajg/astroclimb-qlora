from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .data import (
    DEFAULT_DATASET,
    build_examples,
    load_frame,
    prepare_train_frame,
    split_train_val,
)
from .model import add_lora, load_base_model, load_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AstroCLIMB QLoRA training")
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--train-csv", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="outputs/astroclimb")
    parser.add_argument("--val-per-class", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, default=2560)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--hub-repo-id", default="")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--resume", default="auto")
    return parser.parse_args()


def auto_batch(vram_gb: float) -> tuple[int, int]:
    if vram_gb >= 170:
        return 32, 1
    if vram_gb >= 75:
        return 2, 8
    if vram_gb >= 45:
        return 8, 2
    return 4, 4


def resolve_resume(output_dir: str, resume: str) -> str | None:
    if resume == "none":
        return None
    if resume != "auto":
        return resume
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"), key=lambda path: path.stat().st_mtime
    )
    return str(checkpoints[-1]) if checkpoints else None


class HubPushCallback(TrainerCallback):
    def __init__(self, repo_id: str, private: bool, every: int = 400):
        self.repo_id = repo_id
        self.private = private
        self.every = every

    def on_save(self, args, state, control, **kwargs):
        if not self.repo_id or state.global_step % self.every != 0:
            return control
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(
                repo_id=self.repo_id, private=self.private, exist_ok=True
            )
            folder = f"{args.output_dir}/checkpoint-{state.global_step}"
            api.upload_folder(
                repo_id=self.repo_id,
                folder_path=folder,
                path_in_repo=f"checkpoints/checkpoint-{state.global_step}",
                commit_message=f"checkpoint {state.global_step}",
            )
            print(f"pushed checkpoint {state.global_step}")
        except Exception as error:
            print(f"checkpoint push failed at step {state.global_step}: {error}")
        return control


def make_collator(processor, max_seq_len: int):
    def collate(features):
        texts = []
        images = []
        for feature in features:
            text = (
                processor.apply_chat_template(
                    feature["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                + "\n"
                + feature["label"]
            )
            texts.append(text)
            images.append(list(feature["images"]) if feature["images"] else [])
        if any(images):
            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return collate


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training")
    processor = load_processor(args.model_id)
    frame = prepare_train_frame(load_frame(args.train_csv))
    train_frame, val_frame = split_train_val(
        frame, n_per_class=args.val_per_class, seed=args.seed
    )
    train_examples = build_examples(train_frame, processor)
    val_examples = build_examples(val_frame, processor)
    compute_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    model = load_base_model(args.model_id, compute_dtype=compute_dtype)
    model = prepare_model_for_kbit_training(model)
    model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    batch_size, grad_accum = auto_batch(
        torch.cuda.get_device_properties(0).total_memory / 1e9
    )
    if args.batch_size > 0:
        batch_size = args.batch_size
    if args.grad_accum > 0:
        grad_accum = args.grad_accum
    signature = set(inspect.signature(TrainingArguments.__init__).parameters)
    evaluation_key = (
        "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    )
    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": grad_accum,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "bf16": args.precision == "bf16",
        "fp16": args.precision == "fp16",
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "push_to_hub": False,
        "report_to": "none",
        "remove_unused_columns": False,
        evaluation_key: "steps",
        "eval_steps": args.eval_steps,
    }
    if "warmup_ratio" in signature:
        training_kwargs["warmup_ratio"] = 0.05
    elif "warmup_steps" in signature:
        training_kwargs["warmup_steps"] = int(0.05 * args.max_steps)
    training_kwargs = {
        key: value for key, value in training_kwargs.items() if key in signature
    }
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        data_collator=make_collator(processor, args.max_seq_len),
        compute_metrics=None,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=args.threshold,
            ),
            HubPushCallback(args.hub_repo_id, args.private),
        ],
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resolve_resume(args.output_dir, args.resume))
    output = Path(args.output_dir) / "adapter_best"
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    if args.hub_repo_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.hub_repo_id, private=args.private, exist_ok=True
        )
        api.upload_folder(
            repo_id=args.hub_repo_id,
            folder_path=str(output),
            path_in_repo="adapter_best",
            commit_message=f"best eval_loss {trainer.state.best_metric:.4f}",
        )
        print(f"adapter pushed to {args.hub_repo_id}/adapter_best")


if __name__ == "__main__":
    main()
