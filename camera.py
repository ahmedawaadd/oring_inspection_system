"""Camera setup and frame capture for the Pi HQ Camera (IMX477)."""

import time

import cv2

from config import PREVIEW_RESOLUTION


def init_camera():
    """Create, configure, and start the Picamera2, returning the camera handle.

    picamera2 is imported lazily so the rest of the package imports cleanly on
    machines without the Pi camera stack installed."""
    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": PREVIEW_RESOLUTION, "format": "RGB888"}
    ))
    cam.start()
    time.sleep(2)  # give the camera sensor time to settle before capturing
    return cam


def capture_still(cam):
    # The camera outputs RGB but OpenCV works in BGR, so convert on the way in
    return cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
