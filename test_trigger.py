"""Tests for trigger.py — runs entirely with GPIOStub, no hardware required."""

from __future__ import annotations

import threading
import unittest

from config import Config
from trigger import GPIOStub, State, TriggerStateMachine


def _make_sm(config: Config | None = None, gpio: GPIOStub | None = None) -> tuple[TriggerStateMachine, GPIOStub]:
    cfg = config or Config(use_gpio_stub=True)
    stub = gpio or GPIOStub()
    sm = TriggerStateMachine(cfg, gpio=stub)
    sm.setup()
    return sm, stub


class TestGPIOStub(unittest.TestCase):
    def test_simulate_trigger_calls_registered_callback(self) -> None:
        stub = GPIOStub()
        fired: list[int] = []
        stub.add_event_detect(17, stub.RISING, callback=lambda ch: fired.append(ch))
        stub.simulate_trigger(17)
        self.assertEqual(fired, [17])

    def test_simulate_trigger_unknown_pin_is_noop(self) -> None:
        stub = GPIOStub()
        stub.simulate_trigger(99)  # should not raise

    def test_remove_event_detect_unregisters_callback(self) -> None:
        stub = GPIOStub()
        fired: list[int] = []
        stub.add_event_detect(17, stub.RISING, callback=lambda ch: fired.append(ch))
        stub.remove_event_detect(17)
        stub.simulate_trigger(17)
        self.assertEqual(fired, [])

    def test_cleanup_removes_all_callbacks(self) -> None:
        stub = GPIOStub()
        stub.add_event_detect(17, stub.RISING, callback=lambda ch: None)
        stub.add_event_detect(18, stub.RISING, callback=lambda ch: None)
        stub.cleanup()
        # No exception is the success condition
        stub.simulate_trigger(17)
        stub.simulate_trigger(18)


class TestStateMachineInitial(unittest.TestCase):
    def test_initial_state_is_idle(self) -> None:
        sm, _ = _make_sm()
        self.assertEqual(sm.state, State.IDLE)
        sm.cleanup()


class TestTriggerTransitions(unittest.TestCase):
    def setUp(self) -> None:
        cfg = Config(use_gpio_stub=True, trigger_pin=17)
        self.sm, self.gpio = _make_sm(cfg)

    def tearDown(self) -> None:
        self.sm.cleanup()

    def test_trigger_transitions_idle_to_triggered(self) -> None:
        self.gpio.simulate_trigger(17)
        self.assertEqual(self.sm.state, State.TRIGGERED)

    def test_trigger_ignored_when_not_idle(self) -> None:
        self.sm.advance(State.CAPTURING)
        self.gpio.simulate_trigger(17)
        self.assertEqual(self.sm.state, State.CAPTURING)

    def test_wait_for_trigger_returns_true_when_fired(self) -> None:
        def fire_after_delay() -> None:
            import time
            time.sleep(0.05)
            self.gpio.simulate_trigger(17)

        threading.Thread(target=fire_after_delay, daemon=True).start()
        result = self.sm.wait_for_trigger(timeout=1.0)
        self.assertTrue(result)

    def test_wait_for_trigger_returns_false_on_timeout(self) -> None:
        result = self.sm.wait_for_trigger(timeout=0.05)
        self.assertFalse(result)

    def test_wait_clears_event_so_second_wait_does_not_immediately_return(self) -> None:
        self.gpio.simulate_trigger(17)
        self.sm.wait_for_trigger(timeout=0.1)
        self.sm.advance(State.IDLE)
        second = self.sm.wait_for_trigger(timeout=0.05)
        self.assertFalse(second)


class TestManualStateTransitions(unittest.TestCase):
    def setUp(self) -> None:
        self.sm, self.gpio = _make_sm()

    def tearDown(self) -> None:
        self.sm.cleanup()

    def test_full_cycle_via_advance(self) -> None:
        self.gpio.simulate_trigger(17)
        self.assertEqual(self.sm.state, State.TRIGGERED)
        for expected in [State.CAPTURING, State.INSPECTING, State.RESULT, State.IDLE]:
            self.sm.advance(expected)
            self.assertEqual(self.sm.state, expected)

    def test_new_trigger_allowed_after_return_to_idle(self) -> None:
        self.gpio.simulate_trigger(17)
        for s in [State.CAPTURING, State.INSPECTING, State.RESULT, State.IDLE]:
            self.sm.advance(s)
        self.gpio.simulate_trigger(17)
        self.assertEqual(self.sm.state, State.TRIGGERED)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_state_reads_do_not_raise(self) -> None:
        sm, gpio = _make_sm()
        errors: list[Exception] = []

        def reader() -> None:
            for _ in range(200):
                try:
                    _ = sm.state
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sm.cleanup()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
