import pytest

from app.runners.common import frames_for_duration, split_text, video_dimensions


@pytest.mark.parametrize(
    "aspect, expected",
    [("9:16", (544, 960)), ("16:9", (960, 544)), ("1:1", (704, 704))],
)
def test_video_dimensions_are_multiples_of_32(aspect, expected):
    width, height = video_dimensions(aspect, 960)
    assert (width, height) == expected
    assert width % 32 == 0 and height % 32 == 0


@pytest.mark.parametrize("seconds", [0.5, 1, 3.3, 5, 8])
def test_frames_are_8k_plus_1_and_close_to_requested(seconds):
    frames = frames_for_duration(seconds, 24)
    assert (frames - 1) % 8 == 0
    assert frames >= 9
    assert abs(frames - seconds * 24) <= 8 or seconds < 1


def test_split_text_keeps_short_text_whole():
    assert split_text("Hello there.", 250) == ["Hello there."]


def test_split_text_respects_limit_and_loses_nothing():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = split_text(text, 120)
    assert all(len(c) <= 120 for c in chunks)
    assert " ".join(chunks) == text


def test_split_text_handles_hindi_danda():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है।"
    assert split_text(text, 20) == ["यह पहला वाक्य है।", "यह दूसरा वाक्य है।"]


def test_split_text_breaks_a_huge_sentence_on_words():
    text = "word " * 200
    chunks = split_text(text, 50)
    assert all(len(c) <= 50 for c in chunks)
    assert " ".join(chunks) == text.strip()


def test_split_text_empty():
    assert split_text("   ", 100) == []
