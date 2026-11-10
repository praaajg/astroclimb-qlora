from __future__ import annotations

from .image import convert_str_to_pil

SYSTEM_PROMPT = """You are an expert in astrophysics figures and captions. Given Object A and Object B (each is either a figure image or a caption text), classify their relation into exactly ONE label based ONLY on what you see/read - no DOI or metadata is provided.

Classes:
- same_figure: The caption directly describes the figure in front of you. Visual elements (axes, labels, numbers, morphology) are mentioned verbatim in the text, or the text reads like "Figure X shows..." matching the image.
- same_paper: Same study, different figures. Similar writing style, same instruments/datasets/authors hinted in text, or visual style (fonts, colors, layout) is consistent, but NOT a direct caption-figure match.
- related_papers: Different papers where one builds on the other. Overlapping methods, shared datasets, or a figure/caption that looks like a cited prior result, but style/authors differ.
- unrelated_papers: No clear link. Different topics, instruments, scales, or writing/visual style with no overlap.

Base your decision only on visual and textual content. Do not assume same_figure is impossible for any pair type - judge from alignment.
Output ONLY the lowercase label (e.g., related_papers), no explanation, no punctuation. Choose exactly one of: same_figure, same_paper, related_papers, unrelated_papers.
"""


def build_user_content(row) -> list[dict]:
    parts = []
    for column, name in [("obj_1", "Object A"), ("obj_2", "Object B")]:
        value = row[column]
        if row[f"{column}_is_img"]:
            parts.append({"name": name, "is_img": True, "pil": convert_str_to_pil(value)})
        else:
            parts.append({"name": name, "is_img": False, "text": str(value)[:2000]})
    return parts


def build_train_messages(row) -> list[dict]:
    parts = build_user_content(row)
    content = [{"type": "text", "text": SYSTEM_PROMPT + "\n"}]
    for part in parts:
        if part["is_img"]:
            image = part["pil"].copy()
            image.thumbnail((448, 448))
            content.append({"type": "image", "image": image})
    for part in parts:
        if not part["is_img"]:
            content.append(
                {
                    "type": "text",
                    "text": "\n" + part["name"] + " (caption): " + part["text"],
                }
            )
        else:
            content.append(
                {"type": "text", "text": "\n" + part["name"] + ": [Figure image]"}
            )
    content.append({"type": "text", "text": "\nAnswer with one label:"})
    return [{"role": "user", "content": content}]


def build_inference_messages(row) -> list[dict]:
    parts = build_user_content(row)
    content = []
    for part in parts:
        if part["is_img"]:
            image = part["pil"].copy()
            image.thumbnail((448, 448))
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": SYSTEM_PROMPT + "\n"})
    for part in parts:
        if not part["is_img"]:
            content.append(
                {
                    "type": "text",
                    "text": "\n" + part["name"] + " (caption): " + part["text"],
                }
            )
    content.append({"type": "text", "text": "\nAnswer with one label:"})
    return [{"role": "user", "content": content}]


def collect_images(messages: list[dict]) -> list:
    return [
        part["image"]
        for turn in messages
        for part in turn["content"]
        if part.get("type") == "image" and "image" in part
    ]
