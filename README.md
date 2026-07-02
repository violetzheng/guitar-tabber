# guitar-tabber

Extract guitar tabs from a guitar solo audio file.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python note_events.py path/to/solo.wav
```

This writes `path/to/solo.notes.json` containing a list of note events:

```json
[
  {
    "onset": 0.01,
    "offset": 0.38,
    "midi_pitches": [40],
    "confidence": 0.59
  }
]
```

- `onset` / `offset` — note start/end time in seconds
- `midi_pitches` — one or more MIDI pitches (more than one means a chord/double-stop)
- `confidence` — model confidence (0-1)

Use `-o`/`--output` to choose a different output path:

```bash
.venv/bin/python note_events.py path/to/solo.wav -o notes.json
```
