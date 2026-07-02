"""Compare extracted note events (notes.json, from note_events.py) against ground
truth (a DadaGP .txt tab, parsed on the fly via tab_reference.py).

Matches notes by onset time + pitch (greedy, closest onset wins), then reports
precision/recall/F1 plus the individual misses so you can see what's going wrong.
"""

from __future__ import annotations

import argparse
import json

from tab_reference import parse_tab_file

ONSET_TOLERANCE_S = 0.10
PITCH_TOLERANCE_SEMITONES = 0


class FlatNote:
    def __init__(self, onset: float, pitch: int, source_index: int, techniques: list[str] | None = None):
        self.onset = onset
        self.pitch = pitch
        self.source_index = source_index
        self.techniques = techniques or []


def flatten(events: list[dict]) -> list[FlatNote]:
    notes = []
    for i, e in enumerate(events):
        for pitch in e["midi_pitches"]:
            notes.append(FlatNote(e["onset"], pitch, i, e.get("techniques")))
    return notes


def match(ref: list[FlatNote], pred: list[FlatNote], onset_tol: float, pitch_tol: int):
    pred_available = [True] * len(pred)
    matches = []
    misses = []

    for r in sorted(ref, key=lambda n: n.onset):
        best_idx, best_diff = None, None
        for i, p in enumerate(pred):
            if not pred_available[i]:
                continue
            if abs(p.onset - r.onset) > onset_tol:
                continue
            if abs(p.pitch - r.pitch) > pitch_tol:
                continue
            diff = abs(p.onset - r.onset)
            if best_diff is None or diff < best_diff:
                best_idx, best_diff = i, diff
        if best_idx is not None:
            pred_available[best_idx] = False
            matches.append((r, pred[best_idx]))
        else:
            misses.append(r)

    extra = [pred[i] for i in range(len(pred)) if pred_available[i]]
    return matches, misses, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tab_path", type=str, help="Path to DadaGP-tokenized .txt ground truth")
    parser.add_argument("notes_json_path", type=str, help="Path to notes.json from note_events.py")
    parser.add_argument("--onset-tolerance", type=float, default=ONSET_TOLERANCE_S)
    parser.add_argument("--pitch-tolerance", type=int, default=PITCH_TOLERANCE_SEMITONES)
    parser.add_argument("--max-examples", type=int, default=10, help="Max misses/extras to print")
    args = parser.parse_args()

    ref_events = [vars(e) for e in parse_tab_file(args.tab_path)]
    with open(args.notes_json_path) as f:
        pred_events = json.load(f)

    ref = flatten(ref_events)
    pred = flatten(pred_events)

    matches, misses, extra = match(ref, pred, args.onset_tolerance, args.pitch_tolerance)

    tp, fn, fp = len(matches), len(misses), len(extra)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Reference notes: {len(ref)}   Predicted notes: {len(pred)}")
    print(f"Matched (TP): {tp}   Missed (FN): {fn}   Extra (FP): {fp}")
    print(f"Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")

    if misses:
        print(f"\nMissed notes (in tab, not detected) — first {args.max_examples}:")
        for n in misses[: args.max_examples]:
            tech = f"  [{', '.join(n.techniques)}]" if n.techniques else ""
            print(f"  t={n.onset:.3f}s  pitch={n.pitch}{tech}")

    if extra:
        print(f"\nExtra notes (detected, not in tab) — first {args.max_examples}:")
        for n in extra[: args.max_examples]:
            print(f"  t={n.onset:.3f}s  pitch={n.pitch}")


if __name__ == "__main__":
    main()
