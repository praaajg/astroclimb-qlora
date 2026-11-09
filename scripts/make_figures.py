"""Regenerate machine-readable results and README figures for AstroCLIMB."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"

LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
DISPLAY = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]

ZERO_SHOT = pd.DataFrame(
    [
        {
            "model": "Gemma-4-E4B",
            "thinking": "no",
            "rows": 1800,
            "macro_f1": 0.4408,
            "accuracy": 0.5206,
        },
        {
            "model": "Qwen3-VL-8B-Thinking",
            "thinking": "yes",
            "rows": 1800,
            "macro_f1": 0.1068,
            "accuracy": 0.2461,
        },
        {
            "model": "SmolVLM-500M",
            "thinking": "no",
            "rows": 1800,
            "macro_f1": 0.1723,
            "accuracy": 0.2511,
        },
        {
            "model": "GLM-4.6V-Flash",
            "thinking": "no",
            "rows": 610,
            "macro_f1": 0.1020,
            "accuracy": 0.2557,
        },
    ]
)

FEW_SHOT = pd.DataFrame(
    [
        {
            "model": "Gemma-4-E4B",
            "thinking": "yes",
            "demos": 4,
            "rows": 1800,
            "macro_f1": 0.3925,
            "accuracy": 0.5050,
        },
        {
            "model": "Qwen3-VL-4B",
            "thinking": "yes",
            "demos": 4,
            "rows": 397,
            "macro_f1": 0.2500,
            "accuracy": 0.3300,
        },
        {
            "model": "GLM-4.1V-9B",
            "thinking": "yes",
            "demos": 4,
            "rows": 445,
            "macro_f1": 0.4200,
            "accuracy": 0.5240,
        },
    ]
)

FINE_TUNED = pd.DataFrame(
    [
        {
            "model": "Gemma-4-E4B",
            "adapter": "LoRA-64",
            "val_loss": 0.3464,
            "official_test_f1": 0.64282,
        },
        {
            "model": "SmolVLM-500M",
            "adapter": "LoRA-32",
            "val_loss": 0.3029,
            "official_test_f1": 0.23675,
        },
    ]
)

PER_CLASS = pd.DataFrame(
    [
        {
            "run": "Gemma-4-E4B",
            "mode": "zero-shot",
            "rows": 1800,
            "same_figure": 0.7760,
            "same_paper": 0.4108,
            "related_papers": 0.0172,
            "unrelated_papers": 0.5593,
        },
        {
            "run": "Gemma-4-E4B",
            "mode": "few-shot",
            "rows": 1800,
            "same_figure": 0.7228,
            "same_paper": 0.2047,
            "related_papers": 0.0505,
            "unrelated_papers": 0.5918,
        },
        {
            "run": "SmolVLM-500M",
            "mode": "zero-shot",
            "rows": 1800,
            "same_figure": 0.0128,
            "same_paper": 0.0088,
            "related_papers": 0.3333,
            "unrelated_papers": 0.3341,
        },
        {
            "run": "Gemma-4-E4B",
            "mode": "fine-tuned",
            "rows": 1800,
            "same_figure": 0.5705,
            "same_paper": 0.5311,
            "related_papers": 0.3508,
            "unrelated_papers": 0.6864,
        },
        {
            "run": "GLM-4.1V-9B",
            "mode": "few-shot",
            "rows": 445,
            "same_figure": 0.8653,
            "same_paper": 0.2571,
            "related_papers": 0.0000,
            "unrelated_papers": 0.5575,
        },
        {
            "run": "Qwen3-VL-4B",
            "mode": "few-shot",
            "rows": 397,
            "same_figure": 0.4709,
            "same_paper": 0.2381,
            "related_papers": 0.0612,
            "unrelated_papers": 0.2286,
        },
    ]
)

VALIDATION_METRICS = {
    "split": {"rows": 1800, "per_class": 450, "seed": 42, "train_rows": 8200},
    "best_checkpoint": {"step": 1000, "val_loss": 0.3464},
    "macro_f1": 0.5347,
    "accuracy": 0.5483,
    "per_class": {
        "same_figure": {"precision": 0.9945, "recall": 0.4000, "f1": 0.5705},
        "same_paper": {"precision": 0.4081, "recall": 0.7600, "f1": 0.5311},
        "related_papers": {"precision": 0.5392, "recall": 0.2600, "f1": 0.3508},
        "unrelated_papers": {"precision": 0.6170, "recall": 0.7733, "f1": 0.6864},
    },
    "error_notes": [
        "256 of 270 missed same_figure pairs were predicted as same_paper.",
        "related_papers errors concentrate in figure-figure pairs: "
        "recall 0.0867; 104 of 150 cases predicted as same_paper.",
    ],
}

TEST_METRICS = {
    "rows": 10000,
    "macro_f1": 0.64282,
    "predicted_class_counts": {
        "same_figure": 684,
        "same_paper": 3005,
        "related_papers": 2058,
        "unrelated_papers": 4253,
    },
    "per_class": {
        "same_figure": {"precision": 0.9386, "recall": 0.6420, "f1": 0.7625},
        "same_paper": {"precision": 0.6453, "recall": 0.6463, "f1": 0.6458},
        "related_papers": {"precision": 0.5573, "recall": 0.3823, "f1": 0.4535},
        "unrelated_papers": {"precision": None, "recall": None, "f1": None},
    },
    "error_notes": [
        "333 of 358 missed same_figure pairs were predicted as same_paper.",
        "Figure-figure pairs remain the hardest subset for related_papers (F1 0.3166).",
    ],
}

PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
}
plt.rcParams.update(PLOT_STYLE)

BLUE = "#2f6feb"
ORANGE = "#e8772e"
GREEN = "#2da44e"
GREY = "#6e7781"


def plot_macro_f1() -> None:
    names = ["Zero-shot\n(validation)", "QLoRA\n(validation)", "QLoRA\n(official test)"]
    values = [0.4408, 0.5347, 0.64282]
    colors = [GREY, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(names, values, color=colors, width=0.55)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            fontsize=10,
        )
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Macro F1")
    ax.set_title("AstroCLIMB: macro F1 across evaluation settings")
    fig.tight_layout()
    fig.savefig(ASSETS / "macro_f1.png", dpi=180)
    plt.close(fig)


def plot_per_class_f1() -> None:
    zero_shot = PER_CLASS[
        (PER_CLASS["run"] == "Gemma-4-E4B") & (PER_CLASS["mode"] == "zero-shot")
    ].iloc[0]
    fine_tuned = PER_CLASS[
        (PER_CLASS["run"] == "Gemma-4-E4B") & (PER_CLASS["mode"] == "fine-tuned")
    ].iloc[0]
    x = np.arange(len(LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    first = ax.bar(
        x - width / 2, [zero_shot[label] for label in LABELS], width, label="Zero-shot", color=GREY
    )
    second = ax.bar(
        x + width / 2,
        [fine_tuned[label] for label in LABELS],
        width,
        label="QLoRA fine-tuned",
        color=BLUE,
    )
    for bars in (first, second):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, DISPLAY)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Validation per-class F1 (1,800 balanced pairs)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "per_class_f1.png", dpi=180)
    plt.close(fig)


def plot_precision_recall() -> None:
    metrics = VALIDATION_METRICS["per_class"]
    x = np.arange(len(LABELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    precision = [metrics[label]["precision"] for label in LABELS]
    recall = [metrics[label]["recall"] for label in LABELS]
    first = ax.bar(x - width / 2, precision, width, label="Precision", color=BLUE)
    second = ax.bar(x + width / 2, recall, width, label="Recall", color=ORANGE)
    for bars in (first, second):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=8,
            )
    ax.set_xticks(x, DISPLAY)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Submitted adapter: validation precision and recall")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "validation_precision_recall.png", dpi=180)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    ZERO_SHOT.to_csv(RESULTS / "zero_shot.csv", index=False)
    FEW_SHOT.to_csv(RESULTS / "few_shot.csv", index=False)
    FINE_TUNED.to_csv(RESULTS / "fine_tuned.csv", index=False)
    PER_CLASS.to_csv(RESULTS / "per_class_f1.csv", index=False)
    (RESULTS / "validation_metrics.json").write_text(
        json.dumps(VALIDATION_METRICS, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / "test_metrics.json").write_text(
        json.dumps(TEST_METRICS, indent=2) + "\n", encoding="utf-8"
    )

    plot_macro_f1()
    plot_per_class_f1()
    plot_precision_recall()
    print(f"Wrote results to {RESULTS} and figures to {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
