"""Playability reward for (string, fret) transitions.
"""

from __future__ import annotations

SHORT_RANGE_FRETS = 4   # frets reachable without repositioning the hand
MAX_FRET = 22


def distance_score(from_fret: int, to_fret: int) -> float:
    """[0, 1] — peaks when frets are identical, drops to 0 beyond short range."""
    fret_dist = abs(to_fret - from_fret)
    if fret_dist <= SHORT_RANGE_FRETS:
        return 1.0 - fret_dist / SHORT_RANGE_FRETS
    return 0.0  # long-range jump: hand must fully reposition regardless of choice


def string_change_score(from_str: int, from_fret: int, to_str: int, to_fret: int) -> float:
    """[0, 1] — penalises cross-string moves, especially same-fret ones."""
    if from_str == to_str:
        return 1.0  # same string: distance rule alone is sufficient
    str_dist = abs(to_str - from_str)
    fret_dist = abs(to_fret - from_fret)
    if fret_dist == 0:
        # Same fret + string change: needs barre or awkward jump
        return 0.5 if str_dist == 1 else 0.2
    # Different fret + string change: adjacent is fine, skipping strings is harder
    return 0.75 if str_dist == 1 else 0.4


def neck_position_score(fret: int) -> float:
    """[0, 1] — prefers lower fret positions.

    Higher frets have smaller physical spacing and more string tension, making
    bends and vibrato harder. This is the Neck Position complexity factor
    (§6.2.7) that the RL agent would otherwise ignore because it only sees
    consecutive transition distances, not absolute fret height.
    """
    return 1.0 - fret / MAX_FRET


def rhythm_scale(time_delta_s: float) -> float:
    """[0.5, 1.0] — fast notes demand tighter hand position, scaling penalties up."""
    clamped = max(0.1, min(time_delta_s, 0.5))
    return 0.5 + 0.5 * (clamped - 0.1) / 0.4


def transition_reward(
    from_str: int,
    from_fret: int,
    to_str: int,
    to_fret: int,
    time_delta_s: float,
) -> float:
    """Combined playability reward for one note transition. Range: [0, 1].

    Weights (sum to 1.0):
      0.50 distance     — fret-jump size between consecutive notes
      0.30 string_change — cross-string move quality
      0.20 neck_position — absolute fret height on the destination note
    All three are then scaled by rhythm (faster notes = harsher penalty for bad choices).
    """
    d = distance_score(from_fret, to_fret)
    s = string_change_score(from_str, from_fret, to_str, to_fret)
    n = neck_position_score(to_fret)
    r = rhythm_scale(time_delta_s)
    return (0.5 * d + 0.3 * s + 0.2 * n) * r
