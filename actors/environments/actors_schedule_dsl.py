import random
import re

# --------------------------------------------------------------------------- #
#                         DSL: Parsing & Expansion                            #
# --------------------------------------------------------------------------- #
# Grammar fragments we support (whitespace ignored):
#   token        := "*" | NAME | "(" NAME ("/" NAME)+ ")"
#   repeater     := "*" INT | "*(" INT ".." INT ")"
#   piece        := token [repeater]?
#   spec         := piece ("->" piece)*
# --------------------------------------------------------------------------- #
_TOKEN_RGX = re.compile(r"\(|\)|\*|[^\s()*\/->]+")  # low‑level tokens, not final split


def _split_pieces(spec: str) -> list[str]:
    """Split `a -> b*(1..5) -> (x/y)` into ['a', 'b*(1..5)', '(x/y)']."""
    return [s.strip() for s in spec.split("->") if s.strip()]


def _parse_piece(piece: str) -> tuple[str | list[str], tuple[int, int] | None]:
    """
    Parse a single grammar piece into its candidate names and optional repeat
    range.

    Returns (candidate_names or '*', repeat_range) where:
      - candidate_names: list[str] if (a/b/..), or '*' (wildcard), or 'name' (literal)
      - repeat_range: None for a single use, or (min_rep, max_rep)
    """
    # Extract repeater
    repeat_match = re.search(r"\*\s*(\((\d+)\.\.(\d+)\)|(\d+))\s*$", piece)
    repeat_range: tuple[int, int] | None = None
    if repeat_match:
        if repeat_match.group(4):
            # *N
            n = int(repeat_match.group(4))
            repeat_range = (n, n)
        else:
            # *(a..b)
            a, b = int(repeat_match.group(2)), int(repeat_match.group(3))
            if a > b:
                raise ValueError(f"Invalid repeat range in '{piece}'")
            repeat_range = (a, b)
        piece = piece[: repeat_match.start()].strip()

    # token extraction
    if piece == "*":
        return "*", repeat_range

    if piece.startswith("(") and piece.endswith(")"):
        # (a/b/c)
        names = re.split(r"[\/|]", piece[1:-1].strip())
        names = [n.strip() for n in names if n.strip()]
        if not names:
            raise ValueError(f"Empty choice group: {piece}")
        return names, repeat_range

    # literal
    return piece, repeat_range


def _expand_piece_for_row(
    candidate: str | list[str],
    repeat_range: tuple[int, int] | None,
    all_names: list[str],
) -> list[str]:
    """
    Expand one piece for a single batch row:

      candidate = 'name' or '*' or [names]
      repeat_range = None or (lo, hi)

    Returns a list of actor names (length = repeats) for this row.
    """
    if repeat_range is None:
        repeats = 1
    else:
        repeats = random.randint(*repeat_range)

    if candidate == "*":
        return [random.choice(all_names) for _ in range(repeats)]
    if isinstance(candidate, list):
        return [random.choice(candidate) for _ in range(repeats)]
    # single literal
    return [candidate] * repeats


def _sample_schedule_for_batch_row(
    spec_pieces: list[tuple[str | list[str], tuple[int, int] | None]],
    all_names: list[str],
) -> list[str]:
    """Sample the sequence of actor names for a single batch row."""
    turns: list[str] = []
    for cand, rep in spec_pieces:
        turns.extend(_expand_piece_for_row(cand, rep, all_names))
    return turns


# --------------------------------------------------------------------------- #
#                         DSL: Sampling Schedules                             #
# --------------------------------------------------------------------------- #


def sample_schedule(spec: str, all_names: list[str]) -> list[str]:
    """
    Sample a schedule from the DSL spec string.

    Args:
        spec: DSL specification string.
        all_names: List of all possible actor names.

    Returns:
        A list of sampled actor names for the schedule.
    """
    pieces = _split_pieces(spec)
    parsed_pieces = [_parse_piece(piece) for piece in pieces]
    return _sample_schedule_for_batch_row(parsed_pieces, all_names)


# --------------------------------------------------------------------------- #
