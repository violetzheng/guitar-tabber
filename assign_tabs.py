"""Run a trained RL policy on a notes.json file to produce string/fret tab assignments.

Outputs a JSON file with one entry per note: onset, offset, midi_pitch, string, fret.
Pass --tab to also print a simple ASCII tab to stdout and save a PDF.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from guitar_env import GuitarTabEnv, MAX_CANDIDATES, MAX_FRET
from rl_policy import ActorCritic

STRING_LABELS = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}

_LEGATO_MAX_GAP_S = 0.12   # onset-to-onset; same string within this → HO or PO
_SLIDE_MAX_GAP_S  = 0.40   # onset-to-onset; same string within this → slide
_SLIDE_MIN_FRETS  = 2      # minimum fret distance to call something a slide

_TECH_CHAR = {"hammer_on": "h", "pull_off": "p", "slide_up": "/", "slide_down": "\\"}


def label_techniques(assignments: list[dict]) -> list[dict]:
    """Return a copy of assignments with a 'technique' key added to each entry.

    The technique describes how note i connects TO note i+1 (last note is always
    'normal').  Labeling is purely heuristic: same string + timing + fret direction.
    """
    result = [dict(a, technique="normal") for a in assignments]
    for i in range(len(result) - 1):
        curr, nxt = result[i], result[i + 1]
        if curr["string"] != nxt["string"]:
            continue
        gap = nxt["onset"] - curr["onset"]
        fret_diff = nxt["fret"] - curr["fret"]
        if fret_diff == 0:
            continue
        if gap <= _LEGATO_MAX_GAP_S:
            curr["technique"] = "hammer_on" if fret_diff > 0 else "pull_off"
        elif gap <= _SLIDE_MAX_GAP_S and abs(fret_diff) >= _SLIDE_MIN_FRETS:
            curr["technique"] = "slide_up" if fret_diff > 0 else "slide_down"
    return result


def _low_fret_logit_bias(env, strength: float = 30.0) -> np.ndarray:
    """Logit bonus that prefers low non-open frets.

    Fret 0 (open string) gets no bonus — treated the same as the highest fret —
    so a fretted position on a higher string is always preferred over an open string
    when both are available.  Frets 1–MAX_FRET scale linearly from strength→0.

    Strength is calibrated to reliably overcome the policy's observed ~4–5 nat
    preference for the first (highest-fret) candidate even when competing with
    the same-string bonus (max 0.5 nat).
    """
    bias = np.zeros(MAX_CANDIDATES, dtype=np.float32)
    for i, (_, fret) in enumerate(env.candidates[env.pos]):
        if fret > 0:
            bias[i] = strength * (1.0 - fret / MAX_FRET)
    return bias


def _same_string_logit_bias(env) -> np.ndarray | None:
    """Logit bonus (per candidate slot) that nudges toward the previous string.

    Scaled by same_string_urgency so the bias is strongest for fast passages
    and zero when there is plenty of time to change strings.
    """
    from reward import same_string_urgency
    if env.prev_str == 0 or env.pos == 0:
        return None
    delta = env.notes[env.pos]["onset"] - env.notes[env.pos - 1]["onset"]
    urgency = same_string_urgency(delta)
    if urgency <= 0:
        return None
    bias = np.zeros(MAX_CANDIDATES, dtype=np.float32)
    for i, (s, _) in enumerate(env.candidates[env.pos]):
        if s == env.prev_str:
            bias[i] = 0.5 * urgency  # tiebreaker only; low-fret bias dominates
    return bias


def assign(notes_json: str, policy_path: str, downtune: int = 0) -> list[dict]:
    env = GuitarTabEnv.from_json(notes_json, downtune=downtune)

    policy = ActorCritic()
    policy.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
    policy.eval()

    obs, valid_mask = env.reset()
    done = False
    while not done:
        bias = _low_fret_logit_bias(env)
        same_str = _same_string_logit_bias(env)
        if same_str is not None:
            bias += same_str
        action, _, _ = policy.act(obs, valid_mask, logit_bias=bias)
        obs, _, done, valid_mask = env.step(action)

    return label_techniques(env.assignments)


def _build_tab_columns(assignments: list[dict]) -> list[tuple]:
    """Convert assignments into a flat column list with timing-proportional spacing.

    Uses the 10th-percentile inter-note gap as one column unit so that a handful
    of very close ornamental notes don't compress the whole tab.  Long rests are
    capped at MAX_SPACER columns so silences don't scroll forever.

    Each element is either:
      ("note", string_num, fret)  — a sounding note
      ("space",)                  — one dash-wide spacer
    """
    if not assignments:
        return []

    onsets = [n["onset"] for n in assignments]
    gaps = [b - a for a, b in zip(onsets, onsets[1:])]

    meaningful = sorted(g for g in gaps if g > 0.03)
    if meaningful:
        unit = meaningful[max(0, len(meaningful) // 10)]
    else:
        unit = 0.1
    unit = max(unit, 0.05)  # floor at 50 ms = 1 column

    MAX_SPACER = 8

    columns: list[tuple] = []
    for i, note in enumerate(assignments):
        if i > 0:
            n_spacers = min(round(gaps[i - 1] / unit) - 1, MAX_SPACER)
            for _ in range(max(n_spacers, 0)):
                columns.append(("space",))
        columns.append(("note", note["string"], note["fret"]))
        tech = note.get("technique", "normal")
        if tech != "normal" and i + 1 < len(assignments) and assignments[i + 1]["string"] == note["string"]:
            columns.append(("technique", note["string"], _TECH_CHAR[tech]))

    return columns


def _columns_to_string_bodies(columns: list[tuple]) -> dict[int, str]:
    """Render column list into a per-string body string (to be joined by the caller)."""
    cells: dict[int, list[str]] = {s: [] for s in range(1, 7)}
    for col in columns:
        if col[0] == "space":
            for s in range(1, 7):
                cells[s].append("-")
        elif col[0] == "technique":
            _, tech_str, tech_char = col
            for s in range(1, 7):
                cells[s].append(tech_char if s == tech_str else "-")
        else:
            _, note_str, fret = col
            w = len(str(fret))
            for s in range(1, 7):
                cells[s].append(str(fret) if s == note_str else "-" * w)
    return {s: "-".join(cells[s]) for s in range(1, 7)}


def render_ascii_tab(assignments: list[dict]) -> str:
    columns = _build_tab_columns(assignments)
    bodies = _columns_to_string_bodies(columns)
    lines = [f"{STRING_LABELS[s]} |--{bodies[s]}--|" for s in range(1, 7)]
    return "\n".join(lines)


_GP_DURATION_TABLE = [
    (1, 16, False), (2, 8, False), (3, 8, True),
    (4, 4, False),  (6, 4, True),  (8, 2, False),
    (12, 2, True),  (16, 1, False),
]  # (n_sixteenths, Duration.value, isDotted)


def _gp_duration(n_sixteenths: int):
    import guitarpro
    n16, value, dotted = min(_GP_DURATION_TABLE, key=lambda x: abs(x[0] - n_sixteenths))
    d = guitarpro.Duration(value=value)
    d.isDotted = dotted
    return d


def _gp_effect(tech: str):
    import guitarpro
    e = guitarpro.NoteEffect()
    if tech in ("hammer_on", "pull_off"):
        e.hammer = True  # GP uses one flag for both; direction inferred from fret numbers
    elif tech in ("slide_up", "slide_down"):
        e.slides = [guitarpro.SlideType.legatoSlideTo]
    return e


def render_gp_file(assignments: list[dict], gp_path: str, bpm: int = 120) -> None:
    """Write assignments as a Guitar Pro 5 file."""
    import guitarpro

    if not assignments:
        return

    QUARTER_TICKS = 960          # ticks per quarter note in GP format
    TICKS_PER_16TH = QUARTER_TICKS // 4   # 240
    TICKS_PER_MEASURE = QUARTER_TICKS * 4  # 4/4

    sixteenth_s = 60.0 / bpm / 4.0

    def to_grid(t: float) -> int:
        return max(0, int(round(t / sixteenth_s)))

    # (grid_16ths, dur_16ths, string, fret, technique) per note
    grid = []
    for i, a in enumerate(assignments):
        pos = to_grid(a["onset"])
        nxt = to_grid(assignments[i + 1]["onset"]) if i + 1 < len(assignments) else pos + 4
        dur = max(1, min(nxt - pos, 16))
        grid.append((pos, dur, a["string"], a["fret"], a.get("technique", "normal")))

    # Interleave rests
    flat: list[tuple] = []
    cursor = 0
    for pos, dur, s, f, tech in grid:
        if pos > cursor:
            flat.append((cursor, pos - cursor, None, 0, "normal"))
        flat.append((pos, dur, s, f, tech))
        cursor = pos + dur

    # Pad to full measure
    n_measures = max(1, -(-cursor // 16))
    if cursor < n_measures * 16:
        flat.append((cursor, n_measures * 16 - cursor, None, 0, "normal"))

    # Split anything crossing a measure boundary; continuation → rest
    split: list[tuple] = []
    for pos, dur, s, f, tech in flat:
        while dur > 0:
            boundary = ((pos // 16) + 1) * 16
            chunk = min(dur, boundary - pos)
            split.append((pos, chunk, s, f, tech))
            pos += chunk
            dur -= chunk
            s = None

    # Group by measure
    by_measure: dict[int, list] = {}
    for pos, dur, s, f, tech in split:
        by_measure.setdefault(pos // 16, []).append((pos % 16, dur, s, f, tech))

    # Build Song
    song = guitarpro.Song()
    song.tempo = bpm
    song.measureHeaders.clear()

    track = song.tracks[0]
    track.name = "Guitar"
    track.strings = [guitarpro.GuitarString(i + 1, p)
                     for i, p in enumerate([64, 59, 55, 50, 45, 40])]
    track.frets = 22
    track.measures.clear()

    for m in range(n_measures):
        hdr = guitarpro.MeasureHeader()
        hdr.number = m + 1
        hdr.start = QUARTER_TICKS + m * TICKS_PER_MEASURE
        hdr.timeSignature.numerator = 4
        hdr.timeSignature.denominator = guitarpro.Duration(value=4)
        song.measureHeaders.append(hdr)

        measure = guitarpro.Measure(track, hdr)
        track.measures.append(measure)
        voice = measure.voices[0]

        for beat_16th, beat_dur, s, f, tech in by_measure.get(m, [(0, 16, None, 0, "normal")]):
            beat = guitarpro.Beat(voice, start=hdr.start + beat_16th * TICKS_PER_16TH)
            beat.duration = _gp_duration(beat_dur)

            if s is None:
                beat.status = guitarpro.BeatStatus.rest
            else:
                beat.status = guitarpro.BeatStatus.normal
                note = guitarpro.Note(
                    beat=beat, string=s, value=f, type=guitarpro.NoteType.normal,
                )
                note.effect = _gp_effect(tech)
                beat.notes.append(note)

            voice.beats.append(beat)

    guitarpro.write(song, gp_path)


def render_pdf_tab(assignments: list[dict], pdf_path: str) -> None:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    FONT = "Courier"
    FONT_SIZE = 10
    LINE_H = 5 * mm
    MARGIN_LEFT = 20 * mm
    MARGIN_TOP = 20 * mm
    PAGE_W, PAGE_H = landscape(A4)

    c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
    c.setFont(FONT, FONT_SIZE)

    columns = _build_tab_columns(assignments)
    if not columns:
        c.save()
        return

    bodies = _columns_to_string_bodies(columns)
    body_len = len(bodies[1])  # all strings are the same length

    char_w = c.stringWidth("0", FONT, FONT_SIZE)
    label_chars = 5  # e.g. "e |--"
    suffix_chars = 3  # "--|"
    chars_per_row = max(1, int((PAGE_W - 2 * MARGIN_LEFT) / char_w) - label_chars - suffix_chars)

    row_block_h = 6 * LINE_H + 4 * mm
    rows_per_page = max(1, int((PAGE_H - 2 * MARGIN_TOP) / row_block_h))

    slices = [(i, min(i + chars_per_row, body_len)) for i in range(0, body_len, chars_per_row)]

    for page_idx, page_slices in enumerate(
        slices[i : i + rows_per_page] for i in range(0, len(slices), rows_per_page)
    ):
        if page_idx > 0:
            c.showPage()
            c.setFont(FONT, FONT_SIZE)
        y = PAGE_H - MARGIN_TOP
        for start, end in page_slices:
            for str_idx, s in enumerate(range(1, 7)):
                row_text = f"{STRING_LABELS[s]} |--{bodies[s][start:end]}--|"
                c.drawString(MARGIN_LEFT, y - str_idx * LINE_H, row_text)
            y -= row_block_h

    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes_json", help="Path to notes.json from note_events.py")
    parser.add_argument("policy_path", help="Path to trained policy .pt file")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path")
    parser.add_argument("--downtune", type=int, default=0)
    parser.add_argument("--tab", action="store_true", help="Print ASCII tab to stdout and save PDF + GP5")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo for GP5 file (default: 120)")
    args = parser.parse_args()

    assignments = assign(args.notes_json, args.policy_path, args.downtune)

    output_path = args.output or str(Path(args.notes_json).with_suffix(".tabs.json"))
    with open(output_path, "w") as f:
        json.dump(assignments, f, indent=2)
    print(f"Assigned {len(assignments)} notes → {output_path}")

    if args.tab:
        print()
        print(render_ascii_tab(assignments))

        pdf_path = str(Path(args.notes_json).with_suffix(".tabs.pdf"))
        render_pdf_tab(assignments, pdf_path)
        print(f"\nPDF saved → {pdf_path}")

        gp_path = str(Path(args.notes_json).with_suffix(".tabs.gp5"))
        render_gp_file(assignments, gp_path, bpm=args.bpm)
        print(f"GP5 saved → {gp_path}")

        opener = "open" if sys.platform == "darwin" else ("xdg-open" if sys.platform.startswith("linux") else "start")
        subprocess.Popen([opener, pdf_path])


if __name__ == "__main__":
    main()
