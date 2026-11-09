"""Tests for the fixed prompt contract."""

import pytest

from astroclimb.prompt import (
    ANSWER_INSTRUCTION,
    MAX_CAPTION_CHARS,
    SYSTEM_PROMPT,
    FigureObject,
    build_content_blocks,
    build_messages,
)


def test_system_prompt_lists_every_class():
    for label in ("same_figure", "same_paper", "related_papers", "unrelated_papers"):
        assert label in SYSTEM_PROMPT
    assert "no DOI or metadata is provided" in SYSTEM_PROMPT


def test_messages_have_system_and_user_turns():
    messages = build_messages(
        FigureObject(name="A", kind="caption", text="A caption"),
        FigureObject(name="B", kind="caption", text="Another caption"),
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"][-1] == {"type": "text", "text": ANSWER_INSTRUCTION}


def test_images_appear_before_their_text_and_captions_are_truncated():
    long_caption = "x" * (MAX_CAPTION_CHARS + 50)
    blocks = build_content_blocks(
        [
            FigureObject(name="A", kind="figure", image="image-object"),
            FigureObject(name="B", kind="caption", text=long_caption),
        ]
    )
    assert blocks[0] == {"type": "text", "text": "Object A:"}
    assert blocks[1] == {"type": "image", "image": "image-object"}
    assert blocks[2] == {"type": "text", "text": "Object B:"}
    assert blocks[3]["type"] == "text"
    assert len(blocks[3]["text"]) == MAX_CAPTION_CHARS


def test_figure_requires_an_image():
    with pytest.raises(ValueError, match="no image"):
        FigureObject(name="A", kind="figure")


def test_caption_requires_text():
    with pytest.raises(ValueError, match="no text"):
        FigureObject(name="A", kind="caption")


def test_invalid_kind_is_rejected():
    with pytest.raises(ValueError, match="Unsupported object kind"):
        FigureObject(name="A", kind="table", text="x")
