"""Extract music notes from a guitar solo audio file."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import predict

# TF 2.16+'s SavedModel loader can't load basic-pitch's bundled TF model
# (Keras 3 incompatibility), so we pin to the bundled ONNX model instead.
MODEL_PATH = build_icassp_2022_model_path(FilenameSuffix.onnx)

# window size threshold for treating as one chord.
CHORD_ONSET_WINDOW_S = 0.05

# MIDI pitches of the six open strings (low E → high e).
_OPEN_STRING_PITCHES = frozenset({40, 45, 50, 55, 59, 64})
_RESONANCE_SEMITONE_THRESHOLD = 12
_RESONANCE_LOOKAHEAD_S = 0.5  # also catches open strings appearing shortly after a high note ends


@dataclass
class NoteEvent:
    onset: float
    offset: float
    midi_pitches: list[int]
    confidence: float


def extract_note_events(audio_path: str) -> list[NoteEvent]:
    _, _, raw_note_events = predict(audio_path, MODEL_PATH)

    # raw_note_events: [(start_time_s, end_time_s, pitch_midi, amplitude, pitch_bend)]
    raw_note_events = sorted(raw_note_events, key=lambda n: n[0])

    events: list[NoteEvent] = []
    for start, end, pitch, amplitude, _pitch_bend in raw_note_events:
        if events and abs(start - events[-1].onset) <= CHORD_ONSET_WINDOW_S:
            chord = events[-1]
            chord.midi_pitches.append(int(pitch))
            chord.offset = max(chord.offset, end)
            chord.confidence = max(chord.confidence, float(amplitude))
        else:
            events.append(
                NoteEvent(
                    onset=float(start),
                    offset=float(end),
                    midi_pitches=[int(pitch)],
                    confidence=float(amplitude),
                )
            )

    for event in events:
        event.midi_pitches.sort()

    return _filter_resonance_artifacts(events)


def _filter_resonance_artifacts(events: list[NoteEvent]) -> list[NoteEvent]:
    """Drop open-string events that are sympathetic resonances of higher notes.

    Covers both concurrent resonance (higher note still sustaining) and decay
    resonance (higher note ended within RESONANCE_LOOKAHEAD_S seconds before).
    """
    keep = []
    for i, ev in enumerate(events):
        if not all(p in _OPEN_STRING_PITCHES for p in ev.midi_pitches):
            keep.append(ev)
            continue
        is_resonance = any(
            j != i
            and other.onset <= ev.onset
            and other.offset + _RESONANCE_LOOKAHEAD_S >= ev.onset
            and max(other.midi_pitches) - min(ev.midi_pitches) >= _RESONANCE_SEMITONE_THRESHOLD
            for j, other in enumerate(events)
        )
        if not is_resonance:
            keep.append(ev)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=str, help="Path to guitar solo audio file")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON path (defaults to <audio_path>.notes.json)",
    )
    args = parser.parse_args()

    events = extract_note_events(args.audio_path)

    output_path = args.output or str(Path(args.audio_path).with_suffix(".notes.json"))
    with open(output_path, "w") as f:
        json.dump([asdict(e) for e in events], f, indent=2)

    print(f"Extracted {len(events)} note events -> {output_path}")


if __name__ == "__main__":
    main()
