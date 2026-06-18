# O-Ring Inspection System

An automated visual inspection system for O-rings, built for the **Raspberry Pi 5** with the **Pi HQ Camera (IMX477)**. It detects missing or misplaced O-rings on an assembly line using computer vision — no manual checking required.

Two inspection modes are available:

| Mode | File | How it works |
|------|------|--------------|
| **Reference diff** | `oring_inspect.py` | Compare a live photo against a saved "good" reference image. No model needed. |
| **YOLO inference** | `main.py` | Run a trained YOLO object-detection model to find and count O-rings. |

---

## Hardware Requirements

- Raspberry Pi 5
- Raspberry Pi HQ Camera (IMX477 sensor)
- GPIO trigger input (default: BCM pin 17) — e.g. a photoelectric sensor or push button

---

## Installation

```bash
git clone https://github.com/ahmedawaadd/oring_inspection_system.git
cd oring_inspection_system
pip install -r requirements.txt
```

For full hardware deployment, uncomment the optional packages in `requirements.txt`:

```
picamera2>=0.3
RPi.GPIO>=0.7
ultralytics>=8.0   # only needed for YOLO mode
Pillow>=10.0       # only needed to save PNG inspection images
```

---

## Quick Start

### Reference diff mode (`oring_inspect.py`)

The simplest way to get started — no trained model required.

```bash
python oring_inspect.py
```

**Workflow:**

1. Press **`1`** or **`2`** to select an O-ring slot, then click and drag on the live preview to draw a box around the O-ring.
2. Release the mouse — the system takes a full-resolution photo and saves it as the reference for that slot.
3. Place a new part under the camera and press **`SPACE`** to inspect. The system compares both regions to their references and shows **PASS** or **FAIL**.
4. Press **`Q`** to quit.

**On-screen controls:**

| Key | Action |
|-----|--------|
| `1` / `2` | Draw reference region for O-ring slot 1 or 2 |
| `SPACE` | Inspect the current part |
| `Q` | Quit |

Two slider bars let you tune sensitivity live without restarting:
- **Noise filter** — ignore per-pixel differences smaller than this (reduces false fails from dust/lighting)
- **Diff threshold** — the maximum average difference allowed before a slot is marked FAIL

Reference images and region coordinates are saved to disk (`reference_1.jpg`, `roi_1.npy`, etc.) and reloaded automatically on the next run.

---

### YOLO inference mode (`main.py`)

For production use with a trained model.

```bash
python main.py
```

The system waits for a GPIO trigger (e.g. a part arriving on the conveyor), captures a frame, runs YOLO inference, and logs the result. It runs in **stub mode by default** — no camera, GPIO, or model weights are required to test the pipeline.

**To switch from stub to real hardware**, edit the flags in `config.py`:

```python
use_camera_stub: bool = False   # use real Pi HQ Camera
use_gpio_stub:   bool = False   # use real GPIO trigger
use_model_stub:  bool = False   # use real YOLO weights
```

**Inspection state machine:**

```
IDLE → TRIGGERED → CAPTURING → INSPECTING → RESULT → IDLE
```

A part **passes** when exactly `expected_oring_count` O-rings are detected above `confidence_threshold`. Everything else is a **FAIL**, with a reason logged.

---

## Configuration (`config.py`)

All tunable values live in one place:

| Setting | Default | Description |
|---------|---------|-------------|
| `frame_width` / `frame_height` | 4056 × 3040 | Full IMX477 resolution |
| `trigger_pin` | 17 | BCM GPIO pin for the hardware trigger |
| `trigger_debounce_ms` | 50 | Debounce time in milliseconds |
| `model_path` | `models/oring_yolo.pt` | Path to YOLO weights file |
| `confidence_threshold` | 0.75 | Minimum detection confidence to count |
| `expected_oring_count` | 2 | Number of O-rings required for a PASS |
| `roi` | `None` | Optional (x, y, w, h) crop before inference |
| `db_path` | `logs/inspections.db` | SQLite log file |
| `image_save_dir` | `logs/images` | Where to save inspection images |
| `pass_sample_rate` | 0.1 | Fraction of PASS images to keep (0.0–1.0) |

---

## Project Structure

```
oring_inspection_system/
├── oring_inspect.py   # standalone reference-diff inspection tool
├── main.py            # production entry point (YOLO + GPIO trigger)
├── config.py          # all configuration in one dataclass
├── camera.py          # Pi HQ Camera wrapper (with stub)
├── inference.py       # YOLO model wrapper and pass/fail logic (with stub)
├── trigger.py         # GPIO trigger state machine (with stub)
├── logger.py          # SQLite result logger + image saver
├── requirements.txt
└── tests/
    ├── test_capture.py
    ├── test_inference.py
    ├── test_logger.py
    └── test_trigger.py
```

---

## Running Tests

```bash
python -m pytest
```

All tests run in stub mode — no hardware needed.

---

## Swapping in a Real YOLO Model

1. Train a YOLO model on your O-ring dataset (e.g. using [Ultralytics](https://docs.ultralytics.com)).
2. Place the weights file at `models/oring_yolo.pt` (or update `model_path` in `config.py`).
3. Set `use_model_stub = False` in `config.py`.
4. Run `python main.py`.

The `ModelStub` in `inference.py` is clearly marked with a `SWAP POINT` comment to make this transition easy.
