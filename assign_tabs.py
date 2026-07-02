"""Run a trained RL policy on a notes.json file to produce string/fret tab assignments.

Outputs a JSON file with one entry per note: onset, offset, midi_pitch, string, fret.
Pass --tab to also print a simple ASCII tab to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from guitar_env import GuitarTabEnv
from rl_policy import ActorCritic

STRING_LABELS = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}


def assign(notes_json: str, policy_path: str, downtune: int = 0) -> list[dict]:
    env = GuitarTabEnv.from_json(notes_json, downtune=downtune)

    policy = ActorCritic()
    policy.load_state_dict(torch.load(policy_path, map_location="cpu", weights_only=True))
    policy.eval()

    obs, valid_mask = env.reset()
    done = False
    while not done:
        action, _, _ = policy.act(obs, valid_mask)
        obs, _, done, valid_mask = env.step(action)

    return env.assignments


def render_ascii_tab(assignments: list[dict]) -> str:
    """One column per note, no timing quantisation."""
    rows: dict[int, list[str]] = {s: [] for s in range(1, 7)}
    for note in assignments:
        s, f = note["string"], note["fret"]
        fret_str = str(f)
        for string_num in range(1, 7):
            rows[string_num].append(fret_str if string_num == s else "-" * len(fret_str))

    lines = []
    for s in range(1, 7):
        body = "-".join(rows[s])
        lines.append(f"{STRING_LABELS[s]} |--{body}--|")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes_json", help="Path to notes.json from note_events.py")
    parser.add_argument("policy_path", help="Path to trained policy .pt file")
    parser.add_argument("-o", "--output", default=None, help="Output JSON path")
    parser.add_argument("--downtune", type=int, default=0)
    parser.add_argument("--tab", action="store_true", help="Print ASCII tab to stdout")
    args = parser.parse_args()

    assignments = assign(args.notes_json, args.policy_path, args.downtune)

    output_path = args.output or str(Path(args.notes_json).with_suffix(".tabs.json"))
    with open(output_path, "w") as f:
        json.dump(assignments, f, indent=2)
    print(f"Assigned {len(assignments)} notes → {output_path}")

    if args.tab:
        print()
        print(render_ascii_tab(assignments))


if __name__ == "__main__":
    main()
