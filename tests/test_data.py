"""Tests for image decoding, preprocessing, and pair typing."""

import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from astroclimb.data import (
    decode_image,
    detect_object_kind,
    load_pairs,
    pair_type,
    prepare_image,
    truncate_caption,
)


def png_bytes(size=(32, 24), color=(255, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_base64_image():
    payload = base64.b64encode(png_bytes()).decode("ascii")
    image = decode_image(payload)
    assert image.mode == "RGB"
    assert image.size == (32, 24)


def test_decode_data_uri_and_raw_bytes():
    raw = png_bytes()
    assert decode_image(raw).size == (32, 24)
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    assert decode_image(uri).size == (32, 24)


def test_decode_image_from_path(tmp_path):
    path = tmp_path / "figure.png"
    path.write_bytes(png_bytes())
    assert decode_image(str(path)).size == (32, 24)


def test_prepare_image_is_square_448():
    image = prepare_image(base64.b64encode(png_bytes(size=(10, 30))).decode("ascii"))
    assert image.size == (448, 448)
    assert image.mode == "RGB"


def test_truncate_caption():
    assert truncate_caption(None) == ""
    assert truncate_caption("abc", max_chars=2) == "ab"


def test_detect_object_kind():
    payload = base64.b64encode(png_bytes()).decode("ascii")
    assert detect_object_kind(payload) == "figure"
    assert detect_object_kind("data:image/png;base64," + payload) == "figure"
    assert detect_object_kind("figure.png") == "figure"
    assert detect_object_kind("A short scientific caption") == "caption"


def test_pair_type():
    assert pair_type("figure", "caption") == "TXT-IMG"
    assert pair_type("figure", "figure") == "IMG-IMG"
    assert pair_type("caption", "caption") == "TXT-TXT"
    with pytest.raises(ValueError, match="Unsupported object kinds"):
        pair_type("table", "caption")


def test_load_pairs_infers_types_and_ids(tmp_path):
    payload = base64.b64encode(png_bytes()).decode("ascii")
    path = Path(tmp_path) / "pairs.csv"
    pd.DataFrame(
        {
            "object_a": [payload, "A caption"],
            "object_b": ["A caption", "Another caption"],
            "label": ["same_figure", "unrelated_papers"],
        }
    ).to_csv(path, index=False)

    frame = load_pairs(path)
    assert list(frame.columns) == ["id", "object_a", "object_b", "label", "type_a", "type_b"]
    assert frame["type_a"].tolist() == ["figure", "caption"]
    assert frame["type_b"].tolist() == ["caption", "caption"]
    assert frame["id"].tolist() == [0, 1]


def test_load_pairs_requires_object_columns(tmp_path):
    path = Path(tmp_path) / "bad.csv"
    pd.DataFrame({"caption": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_pairs(path)
