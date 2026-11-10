from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def is_image_str(value) -> bool:
    if not isinstance(value, str) or len(value) < 200:
        return False
    return value.strip().startswith("iVBORw0KGgo")


def convert_str_to_pil(value: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")
