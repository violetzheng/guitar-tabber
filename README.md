# guitar-tabber

Extract guitar tabs from a guitar solo audio file.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

**Step 1 — Extract notes from audio**

```bash
.venv/bin/python note_events.py example_data/example1_amp1.wav
```

Writes `example_data/example1_amp1.notes.json`.

**Step 2 — Assign string/fret positions**
using a trained policy
```bash
.venv/bin/python assign_tabs.py example_data/example1_amp1.notes.json policy.pt --tab
```

Writes `example_data/example1_amp1.notes.tabs.json`. `--tab` also prints an ASCII tab and opens a PDF preview. Use `--downtune N` if the recording is tuned down N semitones from standard.

## Training

Train a new policy on a notes JSON file:

```bash
.venv/bin/python train_rl.py example_data/example1_amp1.notes.json
```

Saves `policy.pt`. Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-o` | `policy.pt` | Output path for the saved policy |
| `--iterations` | `300` | Number of PPO training iterations |
