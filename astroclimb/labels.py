from __future__ import annotations

import re

LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
LABEL_PATTERN = re.compile(
    r"(same_figure|same_paper|related_papers|unrelated_papers)", re.IGNORECASE
)


def parse_label(text: str) -> str:
    match = LABEL_PATTERN.search(text.lower())
    return match.group(1).lower() if match else "unrelated_papers"
