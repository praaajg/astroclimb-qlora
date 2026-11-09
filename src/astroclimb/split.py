"""Reproduce the paper's balanced validation split.

The paper holds out 450 examples per class (1,800 total) from the 10,000-row
competition training file using seed 42, leaving 8,200 training rows. The split
is by pair row index with no index overlap. Figures, captions, and source
papers may recur across splits, so validation measures in-distribution pair
generalization rather than generalization to unseen papers.
"""

from __future__ import annotations

import pandas as pd


def stratified_validation_split(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    n_per_class: int = 450,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, validation)`` with a balanced validation set.

    Sampling is without replacement within each class. The returned frames are
    reset to a clean 0..n-1 index, while the split membership itself is derived
    from the original row index so no row can appear in both partitions.
    """

    if label_col not in frame.columns:
        raise ValueError(f"Missing label column: {label_col!r}")
    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")

    counts = frame[label_col].value_counts()
    too_small = counts[counts < n_per_class]
    if not too_small.empty:
        raise ValueError(
            "Some classes have fewer than n_per_class examples: "
            + ", ".join(f"{label}={count}" for label, count in too_small.items())
        )

    validation_indices: list[int] = []
    for _, group in frame.groupby(label_col, sort=True):
        sampled = group.sample(n=n_per_class, random_state=seed)
        validation_indices.extend(sampled.index.tolist())

    validation_indices = sorted(set(validation_indices))
    validation = frame.loc[validation_indices].sample(frac=1.0, random_state=seed)
    train = frame.drop(index=validation_indices)

    return train.reset_index(drop=True), validation.reset_index(drop=True)
