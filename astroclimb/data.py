from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .image import is_image_str
from .labels import LABEL_TO_ID, LABELS
from .prompts import build_inference_messages, build_train_messages, collect_images

DEFAULT_DATASET = "ShivRamSaud/astroclimb_train"
PAIR_TYPES = ["TXT-IMG", "IMG-IMG", "TXT-TXT"]


def add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["label"] = frame[LABELS].idxmax(axis=1)
    frame["label_id"] = frame["label"].map(LABEL_TO_ID)
    return frame


def add_image_flags(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["obj_1_is_img"] = frame["obj_1"].apply(is_image_str)
    frame["obj_2_is_img"] = frame["obj_2"].apply(is_image_str)
    frame["pair_type"] = frame.apply(
        lambda row: ("IMG" if row["obj_1_is_img"] else "TXT")
        + "-"
        + ("IMG" if row["obj_2_is_img"] else "TXT"),
        axis=1,
    )
    return frame


def prepare_train_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return add_image_flags(add_labels(frame))


def prepare_test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return add_image_flags(frame)


def sample_balanced(
    frame: pd.DataFrame, n_per_class: int = 450, seed: int = 42
) -> pd.DataFrame:
    assert n_per_class % 3 == 0
    parts = []
    for label in LABELS:
        subset = frame[frame["label"] == label]
        if label == "same_figure":
            parts.append(
                subset[subset["pair_type"] == "TXT-IMG"].sample(
                    n=n_per_class, random_state=seed
                )
            )
        else:
            per_pair = n_per_class // 3
            for pair_type in PAIR_TYPES:
                group = subset[subset["pair_type"] == pair_type]
                parts.append(group.sample(n=per_pair, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)


def split_train_val(
    frame: pd.DataFrame, n_per_class: int = 450, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = sample_balanced(frame, n_per_class=n_per_class, seed=seed)
    train = frame.drop(val.index).reset_index(drop=True)
    val = val.reset_index(drop=True)
    return train, val


def load_frame(source: str = DEFAULT_DATASET) -> pd.DataFrame:
    path = Path(source)
    if path.is_file():
        return pd.read_csv(path)
    try:
        return load_dataset(source, split="train").to_pandas()
    except Exception:
        csv_path = hf_hub_download(
            repo_id=source, filename="train.csv", repo_type="dataset"
        )
        return pd.read_csv(csv_path)


def row_to_example(row, processor, inference: bool = False) -> dict:
    messages = (
        build_inference_messages(row) if inference else build_train_messages(row)
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=inference
    )
    return {
        "messages": messages,
        "text": text,
        "images": collect_images(messages),
        "label": row.get("label"),
    }


def build_examples(frame: pd.DataFrame, processor, inference: bool = False) -> list[dict]:
    return [
        row_to_example(frame.iloc[index], processor, inference)
        for index in range(len(frame))
    ]
