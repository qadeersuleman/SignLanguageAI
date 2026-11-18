"""Non-blocking Python Text-to-Speech (TTS) engine with multi-backend support."""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from typing import Optional


class VoiceEngine:
    """Thread-safe, non-blocking local voice output engine.

    Supports Linux `spd-say`, `espeak-ng`, `espeak`, and `pyttsx3` fallback.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._speaking = False
        self._last_spoken_text: str = ""
        self._last_spoken_time: float = 0.0
        self.backend: str = self._detect_backend()
        self._pyttsx3_engine = None
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    @staticmethod
    def _detect_backend() -> str:
        if shutil.which("spd-say"):
            return "spd-say"
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("espeak"):
            return "espeak"
        try:
            import pyttsx3  # noqa: F401
            return "pyttsx3"
        except Exception:
            return "none"

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def speak(self, text: str, rate: float = 1.0, force: bool = False) -> bool:
        """Queue text to be spoken asynchronously."""
        clean = text.strip()
        if not clean or self.backend == "none":
            return False

        now = time.time()
        if not force and clean == self._last_spoken_text and (now - self._last_spoken_time) < 1.2:
            return False

        # Drop stale pending utterances so voice stays snappy and responsive
        if force:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        try:
            self._queue.put_nowait((clean, rate))
            self._last_spoken_text = clean
            self._last_spoken_time = now
            return True
        except queue.Full:
            return False

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                text, rate = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            self._speaking = True
            try:
                self._speak_sync(text, rate)
            except Exception:
                pass
            finally:
                self._speaking = False

    def _speak_sync(self, text: str, rate: float) -> None:
        clamped_rate = max(0.6, min(1.8, rate))

        if self.backend == "spd-say":
            # spd-say rate is -100 to +100 (0 is default)
            spd_rate = int((clamped_rate - 1.0) * 60)
            subprocess.run(
                ["spd-say", "--wait", "-r", str(spd_rate), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=12,
                check=False,
            )
            return

        if self.backend in ("espeak-ng", "espeak"):
            wpm = int(165 * clamped_rate)
            subprocess.run(
                [self.backend, "-s", str(wpm), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=12,
                check=False,
            )
            return

        if self.backend == "pyttsx3":
            import pyttsx3

            if self._pyttsx3_engine is None:
                self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", int(165 * clamped_rate))
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()

    def shutdown(self) -> None:
        self._stop_event.set()
