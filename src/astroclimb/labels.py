"""Label definitions and robust parsing for AstroCLIMB generations."""

from __future__ import annotations

import re

LABELS: tuple[str, ...] = (
    "same_figure",
    "same_paper",
    "related_papers",
    "unrelated_papers",
)
DEFAULT_LABEL = "unrelated_papers"

_LABEL_PATTERN = re.compile(
    "|".join(re.escape(label) for label in LABELS),
    flags=re.IGNORECASE,
)


def parse_label(generation: str | None, default: str = DEFAULT_LABEL) -> str:
    """Map a free-form generation to one of the four labels.

    The paper uses a case-insensitive regex over the four class names and falls
    back to ``unrelated_papers`` when no class name is present. This function
    preserves that behaviour while also accepting labels embedded in a longer
    sentence (for example ``"Label: same_paper"``).
    """

    if not generation:
        return default
    match = _LABEL_PATTERN.search(generation)
    return match.group(0).lower() if match else default


def label_to_index(label: str) -> int:
    """Return the deterministic column index for ``label``."""

    try:
        return LABELS.index(label)
    except ValueError as exc:
        raise ValueError(f"Unknown label {label!r}; expected one of {LABELS}") from exc
