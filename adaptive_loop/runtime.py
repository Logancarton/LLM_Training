"""The never-ending loop: a heartbeat process with brain states and an On/Off switch.

States:  awake (a turn in the last 10 min) -> quiet wake (idle) -> sleep (2 h idle) | off
The On/Off flag lives in data/control.json and is changed ONLY by the control panel.
No prompt, tool or LLM output can reach it.
Off stops the loop, not the clock: on resume, decay for the whole gap is applied once.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import config
from brain import Brain


class Control:
    """The On/Off flag. Survives restarts and reboots."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or config.CONTROL_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(True)

    def _write(self, on: bool) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"on": on, "changed_at": time.time()}))
        tmp.replace(self.path)   # atomic: the flag is never half-written

    def is_on(self) -> bool:
        try:
            return bool(json.loads(self.path.read_text()).get("on", False))
        except (OSError, ValueError):
            return False   # unreadable flag fails safe: off

    def set_on(self, on: bool) -> None:
        self._write(on)


class Runtime:
    def __init__(self, brain: Brain, control: Control):
        self.brain = brain
        self.control = control
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_tick_at: Optional[float] = None

    # --- states ---------------------------------------------------------------

    def state(self, now: Optional[float] = None, last: Optional[float] = None) -> str:
        if not self.control.is_on():
            return "off"
        now = now or time.time()
        if last is None:
            last = self.brain.snapshot()["last_turn_at"]
        if last is None:
            return "quiet"
        idle_min = (now - last) / 60.0
        if idle_min < config.AWAKE_MINUTES:
            return "awake"
        if idle_min < config.SLEEP_AFTER_MINUTES:
            return "quiet"
        return "sleep"

    # --- heartbeat --------------------------------------------------------------

    def heartbeat_once(self, now: Optional[float] = None) -> Optional[float]:
        """One tick. Quiet wake and sleep make no LLM calls in v0.
        (v1: sleep will also run replay and consolidation here.)"""
        if not self.control.is_on():
            return None
        minutes = self.brain.tick(now)
        self.last_tick_at = now or time.time()
        return minutes

    def _run(self) -> None:
        while not self._stop.wait(config.HEARTBEAT_SECONDS):
            try:
                self.heartbeat_once()
            except Exception as exc:   # a bad tick must never kill the loop
                print(f"[heartbeat] error: {exc}")

    def start(self) -> None:
        if self.control.is_on():
            self.heartbeat_once()   # catch up on any time that passed while stopped
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # --- the button ---------------------------------------------------------------

    def toggle(self) -> bool:
        if self.control.is_on():
            with self.brain.lock:   # waits for any turn in progress to finish its write
                self.control.set_on(False)
            return False
        self.control.set_on(True)
        self.heartbeat_once()       # resume: apply decay for the whole off period, once
        return True

    def chat(self, text: str) -> dict:
        if not self.control.is_on():
            return {"error": "Sonara is off. Turn her on with the button."}
        text = (text or "").strip()
        if not text:
            return {"error": "Empty message."}
        return self.brain.handle_turn(text)

    def status(self) -> dict:
        snap = self.brain.snapshot()
        snap.update({"on": self.control.is_on(),
                     "state": self.state(last=snap["last_turn_at"]),
                     "last_tick_at": self.last_tick_at})
        return snap
