"""assigns (string, fret) to each note in a notes.json sequence.

State  : [prev_string, prev_fret, K upcoming pitches, K inter-onset deltas, direction]
         = 11 floats (K = 4)
Action : index into the candidate list for the current note (≤ 6 candidates,
         one per string; invalid slots are masked to -inf before sampling)
Reward : playability score for the chosen → next-note transition (see reward.py)
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from reward import transition_reward

STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}  # s1=high e .. s6=low E
MAX_FRET = 22
MAX_CANDIDATES = 6   # fixed action-space width; invalid slots are masked
K = 4                # lookahead window (current note + K-1 future notes)
STATE_DIM = 2 + K + K + 1  # 11


def get_candidates(midi_pitch: int, downtune: int = 0) -> list[tuple[int, int]]:
    """All (string, fret) pairs that produce this pitch, ordered s6→s1 (low→high)."""
    result = []
    for s in range(6, 0, -1):
        fret = midi_pitch - STANDARD_TUNING[s] + downtune
        if 0 <= fret <= MAX_FRET:
            result.append((s, fret))
    return result


class GuitarTabEnv:
    def __init__(self, note_events: list[dict], downtune: int = 0):
        self.downtune = downtune

        # Flatten chords: take the highest pitch (melody note).
        self.notes: list[dict] = []
        for e in note_events:
            pitches = sorted(e.get("midi_pitches", []), reverse=True)
            if not pitches:
                continue
            self.notes.append({
                "onset": float(e["onset"]),
                "offset": float(e["offset"]),
                "midi_pitch": pitches[0],
            })

        # Pre-compute candidates and validity masks for every note.
        self.candidates: list[list[tuple[int, int]]] = [
            get_candidates(n["midi_pitch"], downtune) for n in self.notes
        ]
        self.valid_masks: list[np.ndarray] = []
        for cands in self.candidates:
            mask = np.zeros(MAX_CANDIDATES, dtype=bool)
            mask[: len(cands)] = True
            self.valid_masks.append(mask)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.pos = 0
        self.prev_str = 0    # 0 = no previous assignment (sentinel)
        self.prev_fret = 0
        self.assignments: list[dict] = []
        return self._state(), self.valid_masks[0]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, np.ndarray]:
        cands = self.candidates[self.pos]
        chosen_str, chosen_fret = cands[action]

        if self.prev_str == 0:
            reward = 0.5   # neutral reward for the very first note (no transition)
        else:
            delta = self.notes[self.pos]["onset"] - self.notes[self.pos - 1]["onset"]
            reward = transition_reward(
                self.prev_str, self.prev_fret,
                chosen_str, chosen_fret,
                time_delta_s=max(delta, 0.01),
            )

        self.assignments.append({
            "onset": self.notes[self.pos]["onset"],
            "offset": self.notes[self.pos]["offset"],
            "midi_pitch": self.notes[self.pos]["midi_pitch"],
            "string": chosen_str,
            "fret": chosen_fret,
        })
        self.prev_str = chosen_str
        self.prev_fret = chosen_fret
        self.pos += 1

        done = self.pos >= len(self.notes)
        if done:
            return np.zeros(STATE_DIM, dtype=np.float32), reward, True, np.zeros(MAX_CANDIDATES, dtype=bool)
        return self._state(), reward, False, self.valid_masks[self.pos]

    # ------------------------------------------------------------------
    def _state(self) -> np.ndarray:
        n = self.notes
        i = self.pos

        prev_s = self.prev_str / 6.0
        prev_f = self.prev_fret / MAX_FRET

        # K upcoming pitches (current note first), padded with 0.0
        pitches = [
            n[i + k]["midi_pitch"] / 127.0 if i + k < len(n) else 0.0
            for k in range(K)
        ]

        # K inter-onset deltas, normalised by 2 s, padded with 1.0 (= plenty of time)
        deltas = []
        for k in range(K):
            pos = i + k
            if pos >= len(n) or pos == 0:
                deltas.append(1.0)
            else:
                deltas.append(min((n[pos]["onset"] - n[pos - 1]["onset"]) / 2.0, 1.0))

        # Melodic direction relative to previous note
        if i == 0 or self.prev_str == 0:
            direction = 0.5
        else:
            diff = n[i]["midi_pitch"] - n[i - 1]["midi_pitch"]
            direction = 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5)

        return np.array([prev_s, prev_f] + pitches + deltas + [direction], dtype=np.float32)

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str, downtune: int = 0) -> "GuitarTabEnv":
        with open(path) as f:
            events = json.load(f)
        return cls(events, downtune)
