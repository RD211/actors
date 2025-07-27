from collections.abc import Mapping
from typing import Any


def mask_turns_and_encode(
    tokenizer,
    messages: list[Mapping[str, Any]],
    turn_masks: list[bool] = None,
    mask_non_assistant_turns: bool = False,
) -> tuple[list[int], list[int]]:
    if not turn_masks and not mask_non_assistant_turns:
        turn_masks = [True] * len(messages)
    elif not turn_masks:
        turn_masks = [m["role"] == "assistant" for m in messages]

    size_check = len(messages) == len(turn_masks)
    if not size_check:
        raise ValueError(
            f"Length of messages ({len(messages)}) must match length of turn_masks ({len(turn_masks)})"
        )

    ids_all: list[int] = []
    mask_all: list[int] = []
    prev_ids: list[int] = []

    for i, _ in enumerate(messages):
        full_prefix = messages[: i + 1]
        token_ids = tokenizer.apply_chat_template(
            full_prefix,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors=None,
        )
        if isinstance(token_ids, dict):
            ids = token_ids.get("input_ids")
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            ids = token_ids

        seg = ids[len(prev_ids) :] if prev_ids else ids

        mask_val = 1 if turn_masks[i] else 0

        ids_all.extend(seg)
        mask_all.extend([mask_val] * len(seg))

        prev_ids = ids

    return ids_all, mask_all
