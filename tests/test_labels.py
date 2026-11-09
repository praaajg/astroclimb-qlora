"""Tests for label parsing and the paper's fallback behaviour."""

import pytest

from astroclimb.labels import DEFAULT_LABEL, LABELS, label_to_index, parse_label


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ("same_figure", "same_figure"),
        ("same_paper", "same_paper"),
        ("related_papers", "related_papers"),
        ("unrelated_papers", "unrelated_papers"),
        ("Label: RELATED_PAPERS.", "related_papers"),
        ("I think this is Same_Paper", "same_paper"),
        ("", DEFAULT_LABEL),
        (None, DEFAULT_LABEL),
        ("no class name here", DEFAULT_LABEL),
    ],
)
def test_parse_label(generation, expected):
    assert parse_label(generation) == expected


def test_all_labels_map_to_their_column():
    assert [label_to_index(label) for label in LABELS] == [0, 1, 2, 3]


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="Unknown label"):
        label_to_index("same_document")
