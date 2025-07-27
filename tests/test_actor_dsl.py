from collections import Counter

import pytest

from actors.environments.actors_schedule_dsl import (
    _expand_piece_for_row,
    _parse_piece,
    _sample_schedule_for_batch_row,
    _split_pieces,
    sample_schedule,
)

# ------------------------------------------------------------------ #
#                         Simple Tests                               #
# ------------------------------------------------------------------ #


def test_split_pieces():
    spec = "qwen -> llama -> (gemma/llama/mistral)*(1..5) -> *"
    assert _split_pieces(spec) == [
        "qwen",
        "llama",
        "(gemma/llama/mistral)*(1..5)",
        "*",
    ]


@pytest.mark.parametrize(
    "piece, cand, rng",
    [
        ("qwen", "qwen", None),
        ("*", "*", None),
        ("(a/b/c)", ["a", "b", "c"], None),
        ("qwen*5", "qwen", (5, 5)),
        ("(x/y)*3", ["x", "y"], (3, 3)),
        ("qwen*(1..5)", "qwen", (1, 5)),
        ("(g/l/m)*(2..4)", ["g", "l", "m"], (2, 4)),
    ],
)
def test_parse_piece(piece, cand, rng):
    c, r = _parse_piece(piece)
    assert c == cand
    assert r == rng


def test_expand_piece_for_row():
    all_names = ["a", "b", "c"]
    out = _expand_piece_for_row(["a", "b"], (3, 3), all_names)
    assert len(out) == 3
    assert set(out).issubset({"a", "b"})


def test_sample_schedule_for_batch_row():
    spec_pieces = [
        ("qwen", None),
        (["gemma", "llama", "mistral"], (1, 2)),
        ("gemma", None),
    ]
    all_names = ["qwen", "gemma", "llama", "mistral"]
    turns = _sample_schedule_for_batch_row(spec_pieces, all_names)
    assert turns[0] == "qwen"
    assert turns[-1] == "gemma"
    assert 3 <= len(turns) <= 4
    assert set(turns).issubset(set(all_names))


# ------------------------------------------------------------------ #
#                         Edge Case Tests                            #
# ------------------------------------------------------------------ #


def test_parse_piece_invalid_range_order():
    with pytest.raises(ValueError):
        _parse_piece("x*(5..2)")


def test_parse_piece_empty_group():
    with pytest.raises(ValueError):
        _parse_piece("()")


def test_expand_piece_literal_no_repeat():
    out = _expand_piece_for_row("foo", None, ["foo", "bar"])
    assert out == ["foo"]


def test_expand_piece_group_fixed_repeat():
    out = _expand_piece_for_row(["a", "b"], (2, 2), ["a", "b"])
    assert len(out) == 2
    assert set(out).issubset({"a", "b"})


def test_expand_piece_wildcard_repeat():
    out = _expand_piece_for_row("*", (3, 3), ["x", "y", "z"])
    assert len(out) == 3
    assert set(out).issubset({"x", "y", "z"})


# ------------------------------------------------------------------ #
#                        Integration Tests                           #
# ------------------------------------------------------------------ #


def test_sample_schedule_end_to_end():
    spec = "qwen -> (gemma/llama)*2 -> * -> mistral"
    all_names = ["qwen", "gemma", "llama", "mistral", "falcon"]
    out = sample_schedule(spec, all_names)
    assert out[0] == "qwen"
    assert out[-1] == "mistral"
    assert 4 <= len(out) <= 5
    assert set(out).issubset(set(all_names))


def test_sample_schedule_all_wildcards():
    spec = "* -> * -> *"
    all_names = ["x", "y", "z"]
    out = sample_schedule(spec, all_names)
    assert len(out) == 3
    assert set(out).issubset(set(all_names))


def test_sample_schedule_only_repeaters():
    spec = "(a/b)*(2..3)"
    all_names = ["a", "b"]
    out = sample_schedule(spec, all_names)
    assert 2 <= len(out) <= 3
    assert set(out).issubset(set(all_names))


# ------------------------------------------------------------------ #
#        Statistical Tests: Validate Distribution is Balanced        #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "candidate,repeat_range,expected_keys",
    [
        (["a", "b", "c"], (1, 1), {"a", "b", "c"}),
        (["x", "y"], (2, 2), {"x", "y"}),
        ("*", (3, 3), {"p", "q", "r", "s"}),
    ],
)
def test_distribution_of_random_choices(candidate, repeat_range, expected_keys):
    all_names = list(expected_keys)

    samples = []
    for _ in range(5000):
        samples.extend(_expand_piece_for_row(candidate, repeat_range, all_names))

    counts = Counter(samples)
    total = sum(counts.values())
    freqs = {k: v / total for k, v in counts.items()}

    # Each candidate should appear roughly evenly (10% margin)
    expected_freq = 1 / len(expected_keys)
    for k in expected_keys:
        assert abs(freqs.get(k, 0) - expected_freq) < 0.10, f"{k} frequency off"
