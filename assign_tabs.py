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


def _low_fret_logit_bias(env, strength: float = 30.0) -> np.ndarray:
    """Logit bonus proportional to (1 - fret/MAX_FRET) for each candidate.

    Strength is calibrated to reliably overcome the policy's observed ~4–5 nat
    preference for the first (highest-fret) candidate even when competing with
    the same-string bonus (max 0.5 nat).  At strength 30 the lowest-fret
    candidate always wins unless another candidate is within ~4 frets of it.
    """
    bias = np.zeros(MAX_CANDIDATES, dtype=np.float32)
    for i, (_, fret) in enumerate(env.candidates[env.pos]):
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

    return env.assignments


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

    return columns


def _columns_to_string_bodies(columns: list[tuple]) -> dict[int, str]:
    """Render column list into a per-string body string (to be joined by the caller)."""
    cells: dict[int, list[str]] = {s: [] for s in range(1, 7)}
    for col in columns:
        if col[0] == "space":
            for s in range(1, 7):
                cells[s].append("-")
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
    parser.add_argument("--tab", action="store_true", help="Print ASCII tab to stdout and save PDF")
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
        opener = "open" if sys.platform == "darwin" else ("xdg-open" if sys.platform.startswith("linux") else "start")
        subprocess.Popen([opener, pdf_path])


if __name__ == "__main__":
    main()
