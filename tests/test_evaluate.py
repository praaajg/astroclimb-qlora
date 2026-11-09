"""Tests for macro-F1 evaluation and prediction parsing."""

import pandas as pd
import pytest

from astroclimb.evaluate import compute_metrics, frame_to_labels
from astroclimb.labels import LABELS


def test_perfect_predictions():
    metrics = compute_metrics(LABELS * 3, LABELS * 3)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["n"] == 12
    assert metrics["confusion_matrix"][0][0] == 3


def test_known_confusion():
    y_true = ["same_figure", "same_figure", "same_paper", "unrelated_papers"]
    y_pred = ["same_paper", "same_figure", "same_paper", "related_papers"]
    metrics = compute_metrics(y_true, y_pred)
    per_class = metrics["per_class"]
    assert per_class["same_figure"]["recall"] == pytest.approx(0.5)
    assert per_class["same_paper"]["recall"] == pytest.approx(1.0)
    assert per_class["related_papers"]["recall"] == pytest.approx(0.0)
    assert metrics["confusion_matrix"][0] == [1, 1, 0, 0]
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_frame_to_one_hot_labels():
    frame = pd.DataFrame(
        {
            "same_figure": [1, 0],
            "same_paper": [0, 1],
            "related_papers": [0, 0],
            "unrelated_papers": [0, 0],
        }
    )
    assert frame_to_labels(frame) == ["same_figure", "same_paper"]


def test_frame_to_free_text_labels():
    frame = pd.DataFrame({"prediction": ["Label: related_papers", "unclear"]})
    assert frame_to_labels(frame) == ["related_papers", "unrelated_papers"]


def test_frame_requires_prediction_information():
    with pytest.raises(ValueError, match="needs either"):
        frame_to_labels(pd.DataFrame({"same_figure": [1, 0]}))


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        compute_metrics(["same_figure"], [])
