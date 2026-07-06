# O-Ring Inspection

A live camera tool that checks O-rings for defects by comparing them to a reference photo. Runs on a Raspberry Pi 5 with the Pi HQ Camera (IMX477).

## Requirements

- Raspberry Pi 5
- Pi HQ Camera (IMX477)
- Python dependencies:

```bash
pip install opencv-python numpy picamera2 evdev
```

## How to run

```bash
python main.py
```

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | Entry point and main loop |
| `config.py` | All tunable values, file paths, and colours |
| `camera.py` | Picamera2 setup and frame capture |
| `vision.py` | Image preprocessing and reference comparison |
| `scanner.py` | USB barcode scanner input via evdev |
| `ui.py` | OpenCV drawing (overlay, popup, result banner) and mouse input |
| `storage.py` | Saving/loading references and inspection logs |

To tune behaviour (resolution, thresholds, file paths, scanner settings), edit `config.py`. No other file needs to change.

## How to use

1. Press **`1`** or **`2`**, then click and drag on the live preview to draw a box around an O-ring. Let go, and the camera takes a high-resolution photo and saves it as the reference for that slot.
2. Place a new part under the camera.
3. Press **`SPACE`** to inspect. The tool compares the current view to the reference and shows **PASS** or **FAIL**.
4. Press **`Q`** to quit.

Use the two slider bars to tune sensitivity:
- **Noise filter**: ignore small differences caused by dust or lighting changes
- **Diff threshold**: how different the image has to be before it counts as a FAIL

Reference photos and region coordinates are saved to disk and reloaded automatically next time you run the script.
