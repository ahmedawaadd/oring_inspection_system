"""O-ring inspection system — main entry point.

Wires together all four modules and runs the single-shot inspection loop:

    IDLE → TRIGGERED → CAPTURING → INSPECTING → RESULT → IDLE

Run with:
    python main.py

Both stub mode (default) and real hardware are controlled entirely by
the flags in config.py (use_camera_stub, use_gpio_stub, use_model_stub).
"""

from __future__ import annotations

import logging
import signal
import sys

from config import CONFIG
from camera import make_camera
from inference import InspectionEngine
from logger import InspectionLogger
from trigger import State, TriggerStateMachine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_shutdown_handler(trigger: TriggerStateMachine, camera, inspection_log: InspectionLogger):
    def _handler(sig, frame):  # noqa: ARG001
        log.info("Received signal %d — shutting down cleanly", sig)
        trigger.cleanup()
        camera.stop()
        inspection_log.close()
        sys.exit(0)
    return _handler


def run(config=CONFIG) -> None:
    camera = make_camera(config)
    trigger = TriggerStateMachine(config)
    engine = InspectionEngine(config)
    inspection_log = InspectionLogger(config)

    camera.start()
    trigger.setup()

    shutdown = build_shutdown_handler(trigger, camera, inspection_log)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info(
        "System ready — stub=%s/%s/%s — waiting for trigger on GPIO pin %d",
        "cam" if config.use_camera_stub else "HW",
        "gpio" if config.use_gpio_stub else "HW",
        "model" if config.use_model_stub else "HW",
        config.trigger_pin,
    )

    try:
        while True:
            if not trigger.wait_for_trigger(timeout=1.0):
                continue  # nothing fired; keep polling

            # --- CAPTURING ---
            trigger.advance(State.CAPTURING)
            frame = camera.capture_frame()

            # --- INSPECTING ---
            trigger.advance(State.INSPECTING)
            result = engine.inspect(frame)

            # --- RESULT ---
            trigger.advance(State.RESULT)
            inspection_log.log_result(result, frame)
            verdict = "PASS" if result.passed else "FAIL"
            log.info("Inspection %s — %s", verdict, result.reason)

            # --- back to IDLE ---
            trigger.advance(State.IDLE)

    finally:
        trigger.cleanup()
        camera.stop()
        inspection_log.close()


if __name__ == "__main__":
    run()
