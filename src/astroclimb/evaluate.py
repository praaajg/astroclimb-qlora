"""Macro-F1 evaluation, per-class metrics, and confusion matrices."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .labels import LABELS, parse_label


def compute_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: Sequence[str] = LABELS,
) -> dict[str, object]:
    """Compute the paper's primary metric and supporting diagnostics."""

    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot evaluate an empty prediction set")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))

    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": len(y_true),
        "labels": list(labels),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
    }


def frame_to_labels(frame: pd.DataFrame, labels: Sequence[str] = LABELS) -> list[str]:
    """Extract labels from a free-text or one-hot prediction frame."""

    if "prediction" in frame.columns:
        return [parse_label(value) for value in frame["prediction"]]

    missing = [label for label in labels if label not in frame.columns]
    if missing:
        raise ValueError(
            "Prediction frame needs either a 'prediction' column or all one-hot "
            f"columns; missing {missing}"
        )
    rows = frame[list(labels)].to_numpy()
    return [labels[index] for index in rows.argmax(axis=1)]


def _merge_gold_and_predictions(
    gold: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    id_col: str,
    label_col: str,
) -> pd.DataFrame:
    if id_col in gold.columns and id_col in predictions.columns:
        return gold.merge(
            predictions,
            on=id_col,
            suffixes=("_gold", "_pred"),
            validate="one_to_one",
        )
    if len(gold) != len(predictions):
        raise ValueError(
            "Gold and prediction files have different row counts and no shared id column"
        )
    merged = gold.copy()
    for column in predictions.columns:
        if column not in merged:
            merged[column] = predictions[column].to_numpy()
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate AstroCLIMB predictions")
    parser.add_argument("--gold", required=True, help="CSV with id and gold label columns")
    parser.add_argument(
        "--pred", required=True, help="CSV with id plus one-hot or prediction columns"
    )
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args(argv)

    gold = pd.read_csv(args.gold)
    predictions = pd.read_csv(args.pred)
    merged = _merge_gold_and_predictions(
        gold,
        predictions,
        id_col=args.id_col,
        label_col=args.label_col,
    )

    gold_column = args.label_col if args.label_col in merged.columns else f"{args.label_col}_gold"
    if gold_column not in merged.columns:
        raise ValueError(f"Could not find gold labels in column {args.label_col!r}")

    metrics = compute_metrics(merged[gold_column], frame_to_labels(merged))
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
