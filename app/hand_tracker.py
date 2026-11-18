"""Step 1 & Step 2: Webcam Input + MediaPipe Hand Landmark Detection & Neon Overlay.

Automatically supports MediaPipe Tasks API (HandLandmarker) as well as legacy
mp.solutions.hands, rendering a sleek neon hand skeleton and bounding box on frames.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .ml_model import SignLanguageAIModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe 21-point hand skeleton connections
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}


def ensure_hand_landmarker_model() -> Path:
    """Download the official MediaPipe HandLandmarker task model if missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if HAND_LANDMARKER_PATH.exists() and HAND_LANDMARKER_PATH.stat().st_size > 100_000:
        return HAND_LANDMARKER_PATH
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, str(HAND_LANDMARKER_PATH))
    return HAND_LANDMARKER_PATH


class HandDetectorEngine:
    """Wrapper abstracting MediaPipe Tasks API and legacy Solutions API."""

    def __init__(self) -> None:
        self.mode = "none"
        self._tasks_detector = None
        self._legacy_hands = None
        self._init_detector()

    def _init_detector(self) -> None:
        # Prefer Tasks API on modern MediaPipe (0.10.20+ / 1.x)
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = ensure_hand_landmarker_model()
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self._tasks_detector = mp_vision.HandLandmarker.create_from_options(options)
            self.mode = "tasks"
            return
        except Exception as exc:
            print(f"[HandDetector] Tasks API unavailable ({exc}); trying legacy mp.solutions.hands...")

        # Fallback to legacy mp.solutions.hands
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._legacy_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.55,
            )
            self.mode = "solutions"

    def detect(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """Return list of detected hand landmark arrays, each shaped (21, 3)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.mode == "tasks" and self._tasks_detector is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._tasks_detector.detect(mp_image)
            hands: List[np.ndarray] = []
            if result and result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float64)
                    hands.append(arr)
            return hands

        if self.mode == "solutions" and self._legacy_hands is not None:
            result = self._legacy_hands.process(rgb)
            hands = []
            if result and result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float64)
                    hands.append(arr)
            return hands

        return []

    def close(self) -> None:
        if self._tasks_detector is not None:
            try:
                self._tasks_detector.close()
            except Exception:
                pass
        if self._legacy_hands is not None:
            try:
                self._legacy_hands.close()
            except Exception:
                pass


def draw_neon_hand_overlay(
    frame: np.ndarray,
    hands_landmarks: List[np.ndarray],
    predicted_sign: Optional[str] = None,
) -> np.ndarray:
    """Draw a modern cyber-neon hand skeleton & sleek corner bounding box."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for hand_idx, lm in enumerate(hands_landmarks):
        pts = [(int(pt[0] * w), int(pt[1] * h)) for pt in lm]

        # Bounding box with rounded-feel corner brackets (echoing poster Step 2)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 22
        x1, y1 = max(6, min(xs) - pad), max(6, min(ys) - pad)
        x2, y2 = min(w - 6, max(xs) + pad), min(h - 6, max(ys) + pad)

        box_color = (248, 189, 56) if hand_idx == 0 else (250, 132, 192)  # BGR: Cyan / Violet
        corner_len = max(16, min(32, (x2 - x1) // 4))
        thickness = 2

        # 4 sleek corner brackets
        cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), box_color, thickness, cv2.LINE_AA)
        cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), box_color, thickness, cv2.LINE_AA)

        cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), box_color, thickness, cv2.LINE_AA)
        cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), box_color, thickness, cv2.LINE_AA)

        cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), box_color, thickness, cv2.LINE_AA)
        cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), box_color, thickness, cv2.LINE_AA)

        cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), box_color, thickness, cv2.LINE_AA)
        cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), box_color, thickness, cv2.LINE_AA)

        # Draw soft glow bones
        for a, b in HAND_CONNECTIONS:
            cv2.line(overlay, pts[a], pts[b], (246, 130, 59), 5, cv2.LINE_AA)
            cv2.line(overlay, pts[a], pts[b], (255, 224, 130), 2, cv2.LINE_AA)

        # Draw joints & glowing fingertips
        for idx, (px, py) in enumerate(pts):
            if idx in FINGERTIP_INDICES:
                cv2.circle(overlay, (px, py), 8, (128, 222, 74), -1, cv2.LINE_AA)
                cv2.circle(overlay, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(overlay, (px, py), 5, (248, 189, 56), -1, cv2.LINE_AA)
                cv2.circle(overlay, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Label tag above hand box
        tag = f"Hand {hand_idx + 1}" + (f" · {predicted_sign}" if predicted_sign and hand_idx == 0 else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
        tag_y = max(24, y1 - 8)
        cv2.rectangle(overlay, (x1, tag_y - th - 8), (x1 + tw + 14, tag_y + 4), (24, 17, 10), -1)
        cv2.rectangle(overlay, (x1, tag_y - th - 8), (x1 + tw + 14, tag_y + 4), box_color, 1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            tag,
            (x1 + 7, tag_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    return frame


class VisionPipeline:
    """Background vision worker that continuously captures, detects, and predicts."""

    def __init__(self, ai_model: SignLanguageAIModel, camera_index: int = 0) -> None:
        self.ai_model = ai_model
        self.camera_index = camera_index
        self.detector = HandDetectorEngine()

        self.show_skeleton: bool = True
        self.mirror: bool = True
        self.paused: bool = False

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reconnect_event = threading.Event()

        self._latest_jpeg: Optional[bytes] = None
        self._latest_landmarks: List[np.ndarray] = []
        self._telemetry: Dict = {
            "camera_ok": False,
            "paused": False,
            "fps": 0,
            "hands_count": 0,
            "sign": None,
            "confidence": 0.0,
            "raw_sign": None,
            "probabilities": {},
            "inference_ms": 0.0,
            "stable": False,
            "model_accuracy": round(self.ai_model.model_accuracy * 100, 1),
            "detector_mode": self.detector.mode,
        }

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def request_reconnect(self) -> None:
        self.paused = False
        self._reconnect_event.set()

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_latest_telemetry(self) -> Dict:
        with self._lock:
            return dict(self._telemetry)

    def get_primary_landmarks(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._latest_landmarks:
                return None
            return self._latest_landmarks[0].copy()

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        # Try requested index first, then scan common indices 0..3
        indices = [self.camera_index] + [i for i in range(4) if i != self.camera_index]
        for idx in indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.camera_index = idx
                    return cap
            cap.release()
        return None

    def _make_standby_jpeg(self, message: str) -> bytes:
        canvas = np.zeros((480, 768, 3), dtype=np.uint8)
        canvas[:] = (22, 14, 9)
        cv2.putText(
            canvas,
            message,
            (180, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (248, 189, 56),
            2,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else b""

    def _loop(self) -> None:
        cap = self._open_camera()
        fps_frames = 0
        fps_t0 = time.time()
        current_fps = 0

        while not self._stop_event.is_set():
            if self._reconnect_event.is_set():
                self._reconnect_event.clear()
                if cap is not None:
                    cap.release()
                cap = self._open_camera()

            if self.paused:
                with self._lock:
                    self._telemetry["paused"] = True
                    self._telemetry["hands_count"] = 0
                    self._telemetry["sign"] = None
                time.sleep(0.1)
                continue

            if cap is None or not cap.isOpened():
                standby = self._make_standby_jpeg("Waiting for Webcam...")
                with self._lock:
                    self._latest_jpeg = standby
                    self._latest_landmarks = []
                    self._telemetry.update(
                        {
                            "camera_ok": False,
                            "paused": False,
                            "fps": 0,
                            "hands_count": 0,
                            "sign": None,
                            "confidence": 0.0,
                            "probabilities": {},
                        }
                    )
                time.sleep(1.0)
                cap = self._open_camera()
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            hands = self.detector.detect(frame)
            pred = self.ai_model.predict(hands)

            if self.show_skeleton and hands:
                frame = draw_neon_hand_overlay(frame, hands, pred.get("sign"))

            fps_frames += 1
            now = time.time()
            if now - fps_t0 >= 0.75:
                current_fps = int(round(fps_frames / (now - fps_t0)))
                fps_frames = 0
                fps_t0 = now

            encode_ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if encode_ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self._latest_landmarks = hands
                    self._telemetry = {
                        "camera_ok": True,
                        "paused": False,
                        "fps": current_fps,
                        "hands_count": len(hands),
                        "sign": pred["sign"],
                        "confidence": pred["confidence"],
                        "raw_sign": pred["raw_sign"],
                        "probabilities": pred["probabilities"],
                        "inference_ms": pred["inference_ms"],
                        "stable": pred["stable"],
                        "model_accuracy": round(self.ai_model.model_accuracy * 100, 1),
                        "detector_mode": self.detector.mode,
                    }

            time.sleep(0.01)

        if cap is not None:
            cap.release()
        self.detector.close()

    def shutdown(self) -> None:
        self._stop_event.set()
