# O-Ring Inspection

A live camera tool that checks O-rings for defects by comparing them to a reference photo. Runs on a Raspberry Pi 5 with the Pi HQ Camera (IMX477).

## Requirements

- Raspberry Pi 5
- Pi HQ Camera (IMX477)
- Python dependencies:

```bash
pip install opencv-python numpy picamera2
```

## How to run

```bash
python oring_inspect.py
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

## Code layout

The tool used to live in one big `oring_inspect.py`. It is now split into small modules, each doing one job:

- `oring_inspect.py` - the entry point, just `main()` and the event loop that wires everything together
- `Settings.py` - all the tunable values (resolutions, thresholds, colours, file paths)
- `Vision.py` - cropping, preprocessing and comparing images
- `Inspection.py` - the pass/fail logic that compares a captured still to the references
- `Storage.py` - saving and loading references, ROIs and inspection logs
- `UI.py` - the on-screen overlay, barcode popup, pass/fail flash and mouse handling
- `BarcodeScanner.py` - reads the USB barcode scanner directly via evdev
