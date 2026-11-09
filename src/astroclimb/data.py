"""Dataset utilities for the AstroCLIMB competition release."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .prompt import MAX_CAPTION_CHARS

IMAGE_SIZE = 448
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
_BASE64_IMAGE_PREFIXES = ("iVBORw0KGgo", "/9j/", "R0lGOD", "UklGR")


def truncate_caption(text: str | None, max_chars: int = MAX_CAPTION_CHARS) -> str:
    """Truncate a caption exactly as described in the paper."""

    if not text:
        return ""
    return text[:max_chars]


def decode_image(value: Any) -> Image.Image:
    """Decode a base64 string, raw bytes, path, or PIL image into RGB."""

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
    elif isinstance(value, str):
        if value.startswith("data:image"):
            _, _, value = value.partition(",")
        if Path(value).is_file():
            return Image.open(value).convert("RGB")
        try:
            payload = base64.b64decode(value, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Object is neither an image path nor valid base64") from exc
    else:
        raise TypeError(f"Unsupported image value of type {type(value)!r}")

    try:
        return Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # Pillow raises several subclasses of OSError
        raise ValueError("Could not decode image bytes") from exc


def resize_square(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize an image to the paper's fixed inference resolution."""

    return image.resize((size, size), Image.Resampling.LANCZOS)


def prepare_image(value: Any, size: int = IMAGE_SIZE) -> Image.Image:
    """Decode and resize one figure for the multimodal model."""

    return resize_square(decode_image(value), size=size)


def detect_object_kind(value: Any) -> str:
    """Infer whether a competition cell is a figure or a caption.

    The competition release stores figures as base64-encoded bytes and captions
    as plain text. When explicit type columns are unavailable, this heuristic
    keeps the loader usable without guessing on short strings.
    """

    if isinstance(value, (bytes, bytearray, Image.Image)):
        return "figure"
    if not isinstance(value, str):
        return "caption"
    stripped = value.strip()
    if stripped.startswith("data:image"):
        return "figure"
    if stripped.startswith(_BASE64_IMAGE_PREFIXES):
        return "figure"
    if len(stripped) < 260 and Path(stripped).suffix.lower() in IMAGE_SUFFIXES:
        return "figure"
    return "caption"


def pair_type(kind_a: str, kind_b: str) -> str:
    """Return the task's pair type string (TXT-IMG, IMG-IMG, TXT-TXT)."""

    kinds = {kind_a.lower(), kind_b.lower()}
    if kinds == {"figure"}:
        return "IMG-IMG"
    if kinds == {"caption"}:
        return "TXT-TXT"
    if kinds == {"figure", "caption"}:
        return "TXT-IMG"
    raise ValueError(f"Unsupported object kinds: {kind_a!r}, {kind_b!r}")


def load_pairs(
    csv_path: str | Path,
    *,
    id_col: str = "id",
    object_a_col: str = "object_a",
    object_b_col: str = "object_b",
    label_col: str = "label",
    type_a_col: str = "type_a",
    type_b_col: str = "type_b",
) -> pd.DataFrame:
    """Load and validate a competition-format CSV."""

    frame = pd.read_csv(csv_path)
    required = [object_a_col, object_b_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Preserve optional columns only when they exist so downstream code can
    # operate on both labelled and unlabelled releases.
    if type_a_col not in frame.columns:
        frame[type_a_col] = frame[object_a_col].map(detect_object_kind)
    if type_b_col not in frame.columns:
        frame[type_b_col] = frame[object_b_col].map(detect_object_kind)
    if id_col not in frame.columns:
        frame.insert(0, id_col, range(len(frame)))

    order = [id_col, object_a_col, object_b_col, label_col, type_a_col, type_b_col]
    return frame[[column for column in order if column in frame.columns]].copy()
