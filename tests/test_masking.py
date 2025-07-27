import pytest
from transformers import AutoTokenizer

from actors.environments import mask_turns_and_encode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture()
def base_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am an AI model."},
        {"role": "user", "content": "What's the weather like today?"},
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _segment_lengths(tokenizer, messages):
    lengths = []
    prev = 0
    for i in range(len(messages)):
        ids = tokenizer.apply_chat_template(
            messages[: i + 1],
            add_generation_prompt=False,
            tokenize=True,
            return_tensors=None,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"][0]
        cur = len(ids)
        lengths.append(cur - prev)
        prev = cur
    return lengths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_included(tokenizer, base_messages):
    turn_masks = [True] * len(base_messages)
    ids, masks = mask_turns_and_encode(tokenizer, base_messages, turn_masks)

    assert len(ids) == len(masks), "ids and masks must be same length"
    assert all(m == 1 for m in masks), "No token should be masked out"


def test_masks_apply_per_turn(tokenizer, base_messages):
    turn_masks = [True, True, False, True]
    ids, masks = mask_turns_and_encode(tokenizer, base_messages, turn_masks)

    seg_lens = _segment_lengths(tokenizer, base_messages)

    slices = []
    start = 0
    for ln in seg_lens:
        slices.append(slice(start, start + ln))
        start += ln

    for i, sl in enumerate(slices):
        segment_mask_vals = masks[sl]
        if turn_masks[i]:
            assert all(v == 1 for v in segment_mask_vals), (
                f"Turn {i} should be included but has masked tokens"
            )
        else:
            assert all(v == 0 for v in segment_mask_vals), (
                f"Turn {i} should be masked but has unmasked tokens"
            )


def test_partial_mask_length_mismatch(tokenizer, base_messages):
    bad_turn_masks = [True] * (len(base_messages) - 1)
    with pytest.raises(ValueError):
        _ = mask_turns_and_encode(tokenizer, base_messages, bad_turn_masks)


def test_empty_conversation(tokenizer):
    ids, masks = mask_turns_and_encode(tokenizer, [], [])
    assert ids == [] and masks == [], "Empty conversation must yield empties"
