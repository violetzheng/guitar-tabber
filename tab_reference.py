"""Parse a DadaGP-tokenized GuitarPro export (e.g. example1.txt) into ground-truth
note events, in the same onset/offset/midi_pitches shape note_events.py produces,
so the two can be compared.

Assumptions (validated against example1_amp1.notes.json's first detected note):
  - tick resolution is 960 per quarter note (wait values are multiples of this)
  - strings are numbered s1 (high e) .. s6 (low E), standard tuning
  - downtune:N shifts every open string down N semitones
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

TICKS_PER_QUARTER = 960

# s1 = high e .. s6 = low E, standard tuning open-string MIDI pitches.
STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

NOTE_RE = re.compile(r"^(\w+):note:s(\d+):f(\d+)$")
REST_RE = re.compile(r"^(\w+):rest$")
WAIT_RE = re.compile(r"^wait:(\d+)$")
TEMPO_RE = re.compile(r"^tempo:(\d+)$")
DOWNTUNE_RE = re.compile(r"^downtune:(\d+)$")


@dataclass
class ReferenceNoteEvent:
    onset: float
    offset: float
    midi_pitches: list[int]
    techniques: list[str] = field(default_factory=list)


def parse_tab_file(path: str) -> list[ReferenceNoteEvent]:
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]

    events: list[ReferenceNoteEvent] = []
    tempo_bpm = 120.0
    downtune = 0
    time_s = 0.0

    pending_notes: list[tuple[int, int]] = []  # (string, fret) sounding since current onset
    pending_techniques: list[str] = []
    pending_onset: float | None = None

    def seconds_per_tick() -> float:
        return 60.0 / (tempo_bpm * TICKS_PER_QUARTER)

    def flush(duration_s: float) -> None:
        nonlocal pending_notes, pending_techniques, pending_onset
        if pending_notes:
            pitches = sorted(STANDARD_TUNING[s] - downtune + f for s, f in pending_notes)
            events.append(
                ReferenceNoteEvent(
                    onset=pending_onset,
                    offset=pending_onset + duration_s,
                    midi_pitches=pitches,
                    techniques=pending_techniques,
                )
            )
        pending_notes = []
        pending_techniques = []
        pending_onset = None

    for line in lines:
        if m := TEMPO_RE.match(line):
            tempo_bpm = float(m.group(1))
        elif m := DOWNTUNE_RE.match(line):
            downtune = int(m.group(1))
        elif m := NOTE_RE.match(line):
            _track, string, fret = m.group(1), int(m.group(2)), int(m.group(3))
            if pending_onset is None:
                pending_onset = time_s
            pending_notes.append((string, fret))
        elif REST_RE.match(line):
            continue  # duration is consumed by the wait that follows
        elif m := WAIT_RE.match(line):
            ticks = int(m.group(1))
            duration_s = ticks * seconds_per_tick()
            flush(duration_s)
            time_s += duration_s
        elif line.startswith("nfx:") or line.startswith("param:"):
            pending_techniques.append(line)
        elif line in ("new_measure", "start", "end", "unknown") or line.startswith("instrument"):
            continue
        # anything else (unrecognized token) is silently skipped

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tab_path", type=str, help="Path to DadaGP-tokenized .txt file")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON path (defaults to <tab_path>.reference.json)",
    )
    args = parser.parse_args()

    events = parse_tab_file(args.tab_path)

    output_path = args.output or str(Path(args.tab_path).with_suffix(".reference.json"))
    with open(output_path, "w") as f:
        json.dump([asdict(e) for e in events], f, indent=2)

    print(f"Parsed {len(events)} reference note events -> {output_path}")


if __name__ == "__main__":
    main()
