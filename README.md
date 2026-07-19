# guitar-tabber

Transcribes guitar tabs (.pdf, guitar pro formats) from audio (e.g. guitar solo tracks, including multi-instrument recordings with background music).

## Algorithm

Tab assignment can be viewed as a sequential decision problem: given a detected note, choose which (string, fret) position to play it on. An **Actor-Critic** network is trained with **Proximal Policy Optimisation (PPO)** to find positions that minimise hand movement across a solo and maximise general guitar playability. The policy observes the previous string and fret, the next few upcoming pitches, notes timing, and melodic direction. Its reward signal penalises large position jumps and rewards staying on the same string during fast passages. Additional factors such as preferring lower frets and not open strings are incorporated to ensure playability.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

**Step 1 — Extract notes from audio**

```bash
.venv/bin/python note_events.py solo.wav
```

For a **mixed recording** (multi-instrument recordings), add `--separate` to run Demucs source separation first and isolate the guitar track before pitch detection:

```bash
.venv/bin/python note_events.py full_band.wav --separate
```

Writes `<name>.notes.json`. The first run with `--separate` downloads the `htdemucs_6s` model (~100 MB).

**Step 2 — Assign string/fret positions**

```bash
.venv/bin/python assign_tabs.py solo.notes.json policy.pt --tab --bpm 120
```

Writes three output files:

`.notes.tabs.json` Per-note assignments with string, fret, and technique 
`.notes.tabs.pdf` Paginated ASCII tab 
`.notes.tabs.gp5` Guitar Pro 5 file with hammer-ons, pull-offs, and slides

**CLI flags:**

`--tab` Print prettified ASCII tab and export to PDF + GP5 format
`--bpm N` Tempo for the GP5 file (default: 120) 
`--downtune N` Recording is tuned down N semitones from standard 

## Training

Train a new policy on a notes JSON file:
```bash
.venv/bin/python train_rl.py example_data/example1_amp1.notes.json
```

Exports policy to `policy.pt`. Additional options:

`-o` | `policy.pt` | Output path for the saved policy 
`--iterations` | `300` Number of PPO training iterations 
