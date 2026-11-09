"""Tests for the seed-42 balanced validation split."""

import pandas as pd
import pytest

from astroclimb.split import stratified_validation_split


def make_frame(per_class: int = 500) -> pd.DataFrame:
    labels = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
    return pd.DataFrame(
        {
            "id": list(range(per_class * len(labels))),
            "label": [label for label in labels for _ in range(per_class)],
        }
    )


def test_split_is_balanced_and_disjoint():
    frame = make_frame()
    train, validation = stratified_validation_split(frame, n_per_class=450, seed=42)

    assert len(validation) == 1_800
    assert len(train) == 200
    assert validation["label"].value_counts().to_dict() == {
        "same_figure": 450,
        "same_paper": 450,
        "related_papers": 450,
        "unrelated_papers": 450,
    }
    assert set(train["id"]).isdisjoint(set(validation["id"]))
    assert list(validation.index) == list(range(1_800))
    assert list(train.index) == list(range(200))


def test_split_is_reproducible():
    frame = make_frame()
    first = stratified_validation_split(frame, n_per_class=450, seed=42)[1]
    second = stratified_validation_split(frame, n_per_class=450, seed=42)[1]
    pd.testing.assert_frame_equal(first, second)


def test_split_rejects_undersized_classes():
    frame = make_frame(per_class=100)
    with pytest.raises(ValueError, match="fewer than n_per_class"):
        stratified_validation_split(frame, n_per_class=450)
