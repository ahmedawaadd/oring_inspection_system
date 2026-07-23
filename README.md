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

To tune defaults (resolution, file paths, scanner settings, and access-control
keys), edit `config.py`. Live inspection thresholds are controlled and saved
by Production Engineer mode.

## Development and testing

The test suite runs anywhere, no Pi required: hardware modules (`picamera2`, `evdev`) are stubbed and the camera is faked at its capture seam.

```bash
pip install -r requirements-dev.txt
python -m pytest                # full suite, a second or two
python -m pytest --cov=.        # with the 95% coverage gate used in CI
```

Test layout mirrors the source: `tests/test_vision.py` covers the comparison math including threshold boundaries, `tests/test_storage.py` covers persistence round trips and the CSV schema, `tests/test_scanner.py` drives the evdev protocol against fake devices, `tests/test_ui.py` covers the mouse state machine and rendering, and `tests/test_main.py` covers key handling plus an end-to-end draw/inspect/log workflow. CI (`.github/workflows/ci.yml`) lints and tests every push.

## How to use

The application always starts in **Operator mode**. Operators place the part
in frame and scan its barcode; accepting the barcode immediately inspects and
logs the current view. A PASS advances to the next barcode. A FAIL requires
the same barcode before another attempt, so an unresolved part cannot be
silently replaced.

Operator mode cannot see or change references or inspection sensitivity.
Calibration values and sliders exist only in Production Engineer mode.

### Production Engineer mode

1. Press **`/`** to open the Production Engineer login.
2. Enter the username, press **`ENTER`**, enter the password, then press
   **`ENTER`** again. TAB switches fields and ESC cancels.
3. Press **`1`** or **`2`**, then drag a region on the preview to replace that
   reference.
4. Adjust the Noise and Diff sliders. Changes take effect immediately and are
   saved to `calibration.json`.
5. Press **`S`** to scan a barcode for a calibration test.
6. Press **`L`** to log out and return the station to Operator mode.

Set production credentials through the station environment before deployment:

```bash
export ORING_ENGINEER_USERNAME="your-user"
export ORING_ENGINEER_PASSWORD="your-password"
```

Without those variables, a development fallback of `engineer` / `change-me`
is used. References and calibration survive restarts, but authenticated mode
does not: every launch starts safely in Operator mode. Inspection CSV rows
also record the Noise and Diff thresholds used for each verdict.
