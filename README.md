# Gesture-Controlled HUD — Setup & Experiment Guide

## 1. Install

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the live app

```bash
python gesture_hud.py
```

Controls (shown as an overlay while running):

| Key | Action |
|---|---|
| `q` | Quit |
| `l` | Cycle lighting-condition label (normal_indoor → dim_room → bright_sunlight) |
| `b` | Start / stop a benchmark run under the current lighting label |
| `x` | Toggle real OS-level system control on/off (starts **OFF** for safety) |

Every frame is logged to `telemetry.db` → `telemetry_logs` regardless of whether you're
benchmarking. Pressing `b` twice (start, then stop) additionally writes one summary row
to `benchmark_summary` covering that window.

Recognized gestures (tune thresholds at the top of `gesture_hud.py`):

| Gesture | How it's detected | Command | Real system action (when `x` is on) |
|---|---|---|---|
| Pinch | thumb tip ↔ index tip < 0.40 × palm size | SELECT | `Enter` key |
| Open palm | all 4 non-thumb fingers extended | PLAY / PAUSE | media play/pause key |
| Fist | all 4 non-thumb fingers curled, thumb roughly level | recognized, no command | - |
| Point | only index finger extended, other 3 curled | MUTE | volume mute key |
| Peace | index + middle extended, ring + pinky curled | SNAPSHOT | saves current frame to `snapshots/` (works even with system control off) |
| Thumbs up | fist shape + thumb tip clearly above thumb knuckle | VOLUME UP | volume up key |
| Thumbs down | fist shape + thumb tip clearly below thumb knuckle | VOLUME DOWN | volume down key |
| Swipe left/right | wrist moves > 22% of frame width within ~0.45s | PREV / NEXT TRACK | prev/next track key |

A static hand-shape gesture must hold for 3 consecutive frames (~100 ms) before it fires a
command — the debounce state machine that suppresses accidental "Midas touch" triggers.
Swipes are motion-based and use their own short time-window + cooldown instead.

**System control is OFF by default.** Press `x` in the running app to enable it — once on,
confirmed gestures send real key presses to your OS (volume, media, Enter) via `pyautogui`.
Keep it off while you're still tuning gesture thresholds so you don't spam your system volume.

**Debug overlay**: with `DEBUG_GESTURE = True` (default) you'll see live numbers for
fingers-extended count, pinch ratio, and thumb vertical offset — useful for retuning
thresholds for your own hand/camera setup. Set it to `False` before recording your final demo.

## 3. Run the three experiments

1. **Latency Profiling Test** — just use the app normally for a while (500+ frames);
   every frame's capture/inference/total latency is already logged.
2. **Environmental Lighting Benchmark** — for each of the 3 lighting conditions:
   press `l` to select the label, `b` to start, gesture naturally for ~30–60s, `b` to stop.
   Repeat for all three conditions.
3. **False Positive (Midas Touch) Test** — start a benchmark run (`b`), talk and move your
   hands naturally *without* intending to trigger anything for ~2 minutes, then stop (`b`).
   The false-positive count/rate comes out of the same `benchmark_summary` row.

## 4. Generate report charts & stats

```bash
python analyze_telemetry.py
```

This reads `telemetry.db` and produces:
- `latency_profile.png` — capture/inference/total latency over the last 500 frames, with the 60ms line marked
- `lighting_benchmark.png` — avg latency by lighting condition
- Printed stats (mean/p50/p90/p95/p99, false-positive rates per run) to paste into the report

## 5. Deliverables checklist

- [x] `gesture_hud.py` — main vision + UI + database logic
- [ ] `telemetry.db` — generated after you run the app on your machine
- [ ] `Case_Study_Report.pdf` — see the drafted report; fill in your own charts/screenshots
- [ ] `Presentation_Deck.pptx` — ask me to generate this once you have real data/screenshots
