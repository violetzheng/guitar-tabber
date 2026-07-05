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


def same_string_urgency(time_delta_s: float) -> float:
    """[0, 1] — how strongly to prefer same-string for this transition.

    1.0 when Δt ≤ 0.1 s (≈ 16th notes at 150 BPM, physically hard to cross strings),
    0.0 when Δt ≥ 0.5 s (plenty of time to reposition).
    """
    clamped = max(0.1, min(time_delta_s, 0.5))
    return 1.0 - (clamped - 0.1) / 0.4


def transition_reward(
    from_str: int,
    from_fret: int,
    to_str: int,
    to_fret: int,
    time_delta_s: float,
) -> float:
    """Combined playability reward for one note transition. Range: [0, 1].

    Weights shift as a function of inter-note timing (sum always = 1.0):
      urgency=0 (slow, Δt ≥ 0.5 s): distance 0.50 / string_change 0.30 / neck 0.20
      urgency=1 (fast, Δt ≤ 0.1 s): distance 0.20 / string_change 0.60 / neck 0.20

    Fast passages need the picking hand on the same string; weight string_change
    heavily so the policy learns to stay put rather than hopping strings.
    All three components are then scaled by rhythm_scale.
    """
    d = distance_score(from_fret, to_fret)
    s = string_change_score(from_str, from_fret, to_str, to_fret)
    n = neck_position_score(to_fret)
    r = rhythm_scale(time_delta_s)
    u = same_string_urgency(time_delta_s)
    w_d = 0.35 - 0.30 * u   # 0.35 → 0.05
    w_s = 0.30 + 0.30 * u   # 0.30 → 0.60
    w_n = 0.35               # raised from 0.20 to prefer lower fret positions
    return (w_d * d + w_s * s + w_n * n) * r
