from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_model_class():
    try:
        from transformers import AutoModelForMultimodalLM as model_class
    except ImportError:
        from transformers import AutoModelForImageTextToText as model_class
    return model_class


def patch_peft() -> None:
    import peft.tuners.lora.model as lora_model

    if getattr(lora_model.LoraModel._create_and_replace, "_is_patched", False):
        return
    original = lora_model.LoraModel._create_and_replace

    def patched(self, lora_config, adapter_name, target, *args, **kwargs):
        if "Gemma4ClippableLinear" in str(type(target)):
            return
        try:
            return original(self, lora_config, adapter_name, target, *args, **kwargs)
        except ValueError as error:
            if "Gemma4ClippableLinear" in str(error):
                return
            raise

    lora_model.LoraModel._create_and_replace = patched
    lora_model.LoraModel._create_and_replace._is_patched = True


def load_processor(model_id: str):
    try:
        return AutoProcessor.from_pretrained(
            model_id, padding_side="left", trust_remote_code=True
        )
    except TypeError:
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def load_base_model(model_id: str, compute_dtype=torch.bfloat16, max_memory=None):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    kwargs = {}
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    return get_model_class().from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization,
        trust_remote_code=True,
        **kwargs,
    )


def attach_adapter(model, adapter: str, subfolder: str | None = None):
    kwargs = {"subfolder": subfolder} if subfolder else {}
    return PeftModel.from_pretrained(model, adapter, **kwargs)


def add_lora(
    model, r: int = 64, alpha: int = 128, dropout: float = 0.05
):
    patch_peft()
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    return get_peft_model(model, config)
