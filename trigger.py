"""Trigger handler and inspection state machine.

State transitions
-----------------
IDLE → TRIGGERED → CAPTURING → INSPECTING → RESULT → IDLE

The GPIO interrupt fires _on_trigger(), which moves IDLE → TRIGGERED and
sets a threading.Event.  The main loop calls wait_for_trigger() to block
until that event fires, then drives the remaining transitions via advance().

Public API
----------
TriggerStateMachine(config, gpio=None)
    gpio: optional pre-built GPIO object (used in tests to inject GPIOStub)
    .setup()               -- configure pin + interrupt
    .cleanup()             -- remove interrupt + release GPIO
    .state -> State        -- current state (thread-safe read)
    .wait_for_trigger(timeout) -> bool
    .advance(new_state)    -- explicit transition from the main loop

GPIOStub
    Drop-in replacement for RPi.GPIO when no hardware is present.
    .simulate_trigger(pin) -- call from tests to fire the interrupt callback
"""

from __future__ import annotations

import logging
import threading
from enum import Enum, auto
from typing import Optional

from config import Config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE = auto()
    TRIGGERED = auto()
    CAPTURING = auto()
    INSPECTING = auto()
    RESULT = auto()


# ---------------------------------------------------------------------------
# GPIO stub (no hardware required)
# ---------------------------------------------------------------------------

class GPIOStub:
    """Simulates the RPi.GPIO interface.

    SWAP POINT: when real hardware is available, pass gpio=None to
    TriggerStateMachine and it will import RPi.GPIO automatically.
    """

    # Mirror the constants used in the real library
    BCM = "BCM"
    IN = "IN"
    PUD_DOWN = "PUD_DOWN"
    RISING = "RISING"

    def __init__(self) -> None:
        self._callbacks: dict[int, object] = {}

    def setmode(self, mode: str) -> None:  # noqa: ARG002
        pass

    def setup(self, pin: int, direction: str, pull_up_down: Optional[str] = None) -> None:  # noqa: ARG002
        pass

    def add_event_detect(
        self, pin: int, edge: str, callback, bouncetime: int = 50
    ) -> None:
        self._callbacks[pin] = callback

    def remove_event_detect(self, pin: int) -> None:
        self._callbacks.pop(pin, None)

    def cleanup(self) -> None:
        self._callbacks.clear()

    def simulate_trigger(self, pin: int) -> None:
        """Fire the registered interrupt callback for *pin* (test helper)."""
        if pin in self._callbacks:
            self._callbacks[pin](pin)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TriggerStateMachine:
    def __init__(self, config: Config, gpio=None) -> None:
        self._config = config
        self._gpio = gpio if gpio is not None else self._load_gpio()
        self._state = State.IDLE
        self._lock = threading.Lock()
        self._trigger_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        self._gpio.setmode(self._gpio.BCM)
        self._gpio.setup(
            self._config.trigger_pin,
            self._gpio.IN,
            pull_up_down=self._gpio.PUD_DOWN,
        )
        self._gpio.add_event_detect(
            self._config.trigger_pin,
            self._gpio.RISING,
            callback=self._on_trigger,
            bouncetime=self._config.trigger_debounce_ms,
        )
        log.debug("GPIO trigger armed on pin %d", self._config.trigger_pin)

    def cleanup(self) -> None:
        self._gpio.remove_event_detect(self._config.trigger_pin)
        self._gpio.cleanup()

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def advance(self, new_state: State) -> None:
        """Explicit state transition driven by the main loop."""
        with self._lock:
            log.debug("State: %s → %s", self._state.name, new_state.name)
            self._state = new_state

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------

    def wait_for_trigger(self, timeout: Optional[float] = None) -> bool:
        """Block until a trigger arrives.

        Returns True if triggered, False if *timeout* elapsed first.
        """
        fired = self._trigger_event.wait(timeout=timeout)
        if fired:
            self._trigger_event.clear()
        return fired

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_trigger(self, channel: int) -> None:  # noqa: ARG002
        with self._lock:
            if self._state != State.IDLE:
                log.warning(
                    "Trigger on pin %d ignored — state is %s (not IDLE)",
                    self._config.trigger_pin,
                    self._state.name,
                )
                return
            self._state = State.TRIGGERED
        self._trigger_event.set()

    def _load_gpio(self):
        if self._config.use_gpio_stub:
            return GPIOStub()
        import RPi.GPIO as GPIO  # type: ignore[import]  # deferred import
        return GPIO
