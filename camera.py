"""Camera setup and capture for the Pi HQ Camera (IMX477) via Picamera2."""

import time

import cv2
from picamera2 import Picamera2

from config import PREVIEW_RESOLUTION


def create_camera():
    """Configure and start the camera, then wait for the sensor to settle
    so the first captures have stable exposure and white balance."""
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
    ))
    cam.start()
    time.sleep(2)
    return cam


def capture_still(cam):
    """Capture one frame as BGR. The camera outputs RGB but OpenCV works
    in BGR, so convert on the way in."""
    return cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
