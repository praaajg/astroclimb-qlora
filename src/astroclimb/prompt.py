"""The fixed prompt contract used for training, validation, and test.

The system prompt below is reproduced from Appendix A.1 of the paper. The
paper uses one shared descriptive prompt for every pair type and one fixed
chat template for both training and inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

MAX_CAPTION_CHARS = 2000

SYSTEM_PROMPT = """You are an expert in astrophysics figures and captions. Given Object A and Object B (each is either a figure image or a caption text), classify their relation into exactly ONE label based ONLY on what you see/read - no DOI or metadata is provided.
Classes:
- same_figure: The caption directly describes the figure in front of you. Visual elements (axes, labels, numbers, morphology) are mentioned verbatim in the text, or the text reads like "Figure X shows..." matching the image.
- same_paper: Same study, different figures. Similar writing style, same instruments/datasets/authors hinted in text, or visual style (fonts, colors, layout) is consistent, but NOT a direct caption-figure match.
- related_papers: Different papers where one builds on the other. Overlapping methods, shared datasets, or a figure/caption that looks like a cited prior result, but style/authors differ.
- unrelated_papers: No clear link. Different topics, instruments, scales, or writing/visual style with no overlap.
Base your decision only on visual and textual content. Do not assume same_figure is impossible for any pair type - judge from alignment.
Output ONLY the lowercase label (e.g., related_papers), no explanation, no punctuation. Choose exactly one of: same_figure, same_paper, related_papers, unrelated_papers."""

ANSWER_INSTRUCTION = "Answer with one label:"

ObjectKind = Literal["figure", "caption"]


@dataclass(frozen=True)
class FigureObject:
    """One side of an AstroCLIMB pair.

    ``image`` is any object accepted by the model processor (for example a
    ``PIL.Image.Image`` or a path), and is required when ``kind == "figure"``.
    ``text`` is required when ``kind == "caption"`` and is truncated to 2,000
    characters, matching the paper.
    """

    name: str
    kind: ObjectKind
    image: Any | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("figure", "caption"):
            raise ValueError(f"Unsupported object kind: {self.kind!r}")
        if self.kind == "figure" and self.image is None:
            raise ValueError(f"Object {self.name} is a figure but has no image")
        if self.kind == "caption" and self.text is None:
            raise ValueError(f"Object {self.name} is a caption but has no text")


def build_content_blocks(objects: Sequence[FigureObject]) -> list[dict[str, Any]]:
    """Render objects as multimodal chat content blocks.

    Each object is introduced by name. Figure blocks contain the image itself;
    caption blocks contain the (truncated) caption text. The fixed answer
    instruction is the last block, as in the paper.
    """

    blocks: list[dict[str, Any]] = []
    for obj in objects:
        blocks.append({"type": "text", "text": f"Object {obj.name}:"})
        if obj.kind == "figure":
            blocks.append({"type": "image", "image": obj.image})
        else:
            caption = (obj.text or "")[:MAX_CAPTION_CHARS]
            blocks.append({"type": "text", "text": caption})
    blocks.append({"type": "text", "text": ANSWER_INSTRUCTION})
    return blocks


def build_messages(object_a: FigureObject, object_b: FigureObject) -> list[dict[str, Any]]:
    """Build the exact two-turn chat message list for one pair."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_content_blocks([object_a, object_b])},
    ]
