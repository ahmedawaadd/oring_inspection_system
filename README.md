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

## How to use

1. Press **`1`** or **`2`**, then click and drag on the live preview to draw a box around an O-ring. Let go — the camera takes a high-resolution photo and saves it as the reference for that slot.
2. Place a new part under the camera.
3. Press **`SPACE`** to inspect. The tool compares the current view to the reference and shows **PASS** or **FAIL**.
4. Press **`Q`** to quit.

Use the two slider bars to tune sensitivity:
- **Noise filter** — ignore small differences caused by dust or lighting changes
- **Diff threshold** — how different the image has to be before it counts as a FAIL

Reference photos and region coordinates are saved to disk and reloaded automatically next time you run the script.

## Project structure

The tool is split into focused modules:

| File | Responsibility |
|---|---|
| `main.py` | Entry point: window/trackbar setup and the interactive event loop |
| `config.py` | Tunable constants and the colour palette |
| `camera.py` | Pi camera setup and frame capture |
| `image_ops.py` | Cropping, preprocessing, image comparison, thumbnails |
| `inspection.py` | Loading/capturing references and running a comparison pass |
| `overlay.py` | On-screen UI: status bars, barcode popup, result banner |
| `scanner.py` | USB barcode scanner support (read via evdev) |
| `mouse.py` | Mouse state and the region-drawing callback |
| `logger.py` | Saving result images and the per-barcode CSV log |
