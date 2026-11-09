"""Model loading and LoRA configuration for the AstroCLIMB system."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_processor(model_id: str = DEFAULT_MODEL_ID) -> Any:
    """Load the matching auto processor."""

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id)


def load_base_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    double_quant: bool = True,
    compute_dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    device_map: str | dict[str, Any] = "auto",
) -> Any:
    """Load the frozen vision-language backbone.

    The paper loads ``google/gemma-4-E4B-it`` through
    ``AutoModelForMultimodalLM`` with a fallback to
    ``AutoModelForImageTextToText``. Both paths are supported here so the code
    keeps working across Transformers releases.
    """

    import torch
    from transformers import BitsAndBytesConfig

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(compute_dtype)
    if dtype is None:
        raise ValueError(f"Unsupported compute dtype: {compute_dtype!r}")

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=dtype,
        )

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "attn_implementation": attn_implementation,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    try:
        from transformers import AutoModelForMultimodalLM

        model_class = AutoModelForMultimodalLM
    except (ImportError, AttributeError):
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText

    return model_class.from_pretrained(model_id, **kwargs)


def build_lora_config(
    *,
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Return the LoRA configuration reported in the paper."""

    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type,
        target_modules=list(target_modules),
    )


def attach_adapter(model: Any, adapter_path: str, *, trainable: bool = False) -> Any:
    """Attach a saved LoRA adapter to a base model."""

    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
