#!/usr/bin/env python3
"""O-ring inspection script for Raspberry Pi 5 + Pi HQ Camera (IMX477)."""

import argparse
import sys
import time

import cv2
import numpy as np
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GPIO_BUTTON_PIN = 17          # BCM pin number for trigger button
CAMERA_RESOLUTION = (4056, 3040)  # IMX477 full resolution; reduce for speed
BLUR_KERNEL_SIZE = (5, 5)     # Gaussian blur kernel (must be odd x odd)
DIFF_THRESHOLD = 10.0         # Mean pixel diff below this → PASS

REFERENCE_PATH = "reference.jpg"
# ---------------------------------------------------------------------------


def capture_image(camera):
    """Capture a single frame and return it as a BGR numpy array."""
    frame = camera.capture_array()
    # picamera2 returns RGB by default; convert to BGR for OpenCV
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def preprocess(image):
    """Grayscale + Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)
    return blurred


def compare(ref_proc, sample_proc):
    """Return (pass_bool, mean_diff) comparing two preprocessed images."""
    diff = cv2.absdiff(ref_proc, sample_proc)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
    mean_diff = float(np.mean(thresh))
    return mean_diff < DIFF_THRESHOLD, mean_diff


def setup_camera():
    """Initialize and start the picamera2 instance."""
    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(2)  # warm-up
    return cam


def run_reference():
    """Capture one photo and save it as the reference image."""
    print("Capturing reference image…")
    cam = setup_camera()
    try:
        image = capture_image(cam)
    finally:
        cam.stop()

    cv2.imwrite(REFERENCE_PATH, image)
    print(f"Reference saved to {REFERENCE_PATH}")


def run_inspect():
    """Wait for button press, capture, compare, print PASS/FAIL, repeat."""
    import RPi.GPIO as GPIO

    # Load reference
    ref_image = cv2.imread(REFERENCE_PATH)
    if ref_image is None:
        print(f"ERROR: reference image not found at '{REFERENCE_PATH}'", file=sys.stderr)
        print("Run with --reference first.", file=sys.stderr)
        sys.exit(1)

    ref_proc = preprocess(ref_image)

    # GPIO setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    cam = setup_camera()
    print(f"Ready. Press button (GPIO {GPIO_BUTTON_PIN}) to inspect. Ctrl-C to quit.")

    try:
        while True:
            # Block until button pressed (active-low with pull-up)
            GPIO.wait_for_edge(GPIO_BUTTON_PIN, GPIO.FALLING)
            time.sleep(0.05)  # debounce

            image = capture_image(cam)
            sample_proc = preprocess(image)

            passed, mean_diff = compare(ref_proc, sample_proc)
            result = "PASS" if passed else "FAIL"
            print(f"{result}  (mean_diff={mean_diff:.2f}, threshold={DIFF_THRESHOLD})")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cam.stop()
        GPIO.cleanup()


def main():
    parser = argparse.ArgumentParser(description="O-ring inspection tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", action="store_true",
                       help="Capture reference image and exit")
    group.add_argument("--inspect", action="store_true",
                       help="Run inspection loop (requires reference image)")
    args = parser.parse_args()

    if args.reference:
        run_reference()
    else:
        run_inspect()


if __name__ == "__main__":
    main()
