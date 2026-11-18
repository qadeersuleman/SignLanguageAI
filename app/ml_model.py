"""Hybrid AI Sign Language Recognition Model.

Combines a trained scikit-learn RandomForestClassifier on scale- and mirror-invariant
biomechanical hand landmark features with two-hand spatial reasoning ("Please") and
temporal motion analysis ("Good Bye" waving).
"""

from __future__ import annotations

import json
import math
import pickle
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .signs import SIGN_LABELS

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = MODELS_DIR / "gesture_model.pkl"
CUSTOM_SAMPLES_PATH = DATA_DIR / "custom_samples.json"

STATIC_SINGLE_HAND_SIGNS = [
    "Hello",
    "Thank You",
    "Yes",
    "Stop",
    "Go",
    "Peace",
    "Help",
]

FEATURE_NAMES = [
    "idx_ext",
    "mid_ext",
    "rng_ext",
    "pnk_ext",
    "idx_up",
    "mid_up",
    "rng_up",
    "pnk_up",
    "thumb_ext",
    "thumb_up",
    "thumb_above_idx",
    "pinch_dist",
    "spread_im",
    "spread_mr",
    "spread_rp",
    "idx_dx",
    "idx_abs_dy",
    "fist_compact",
    "palm_upright",
]


def _dist2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def extract_hand_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract 19 scale-, position-, and mirror-invariant features from 21 hand landmarks.

    Args:
        landmarks: array of shape (21, 3) or (21, 2) in normalized image coords [0, 1].
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    wrist = lm[0]
    palm_scale = _dist2d(wrist, lm[9]) + 1e-6

    # 1-4: Finger extension ratios (tip distance vs pip distance from wrist)
    finger_defs = [
        (5, 6, 8),    # index
        (9, 10, 12),  # middle
        (13, 14, 16), # ring
        (17, 18, 20), # pinky
    ]
    exts = []
    ups = []
    compact_dists = []
    for mcp_i, pip_i, tip_i in finger_defs:
        d_tip = _dist2d(lm[tip_i], wrist)
        d_pip = _dist2d(lm[pip_i], wrist) + 1e-6
        exts.append(min(2.4, d_tip / d_pip))
        ups.append((lm[pip_i][1] - lm[tip_i][1]) / palm_scale)
        compact_dists.append(_dist2d(lm[tip_i], lm[mcp_i]) / palm_scale)

    # 9-11: Thumb geometry
    thumb_ext = _dist2d(lm[4], lm[17]) / palm_scale
    thumb_up = (lm[2][1] - lm[4][1]) / palm_scale
    thumb_above_idx = (lm[5][1] - lm[4][1]) / palm_scale

    # 12: OK-sign thumb-to-index pinch distance
    pinch_dist = _dist2d(lm[4], lm[8]) / palm_scale

    # 13-15: Inter-finger tip spreads (distinguishes Hello spread vs Stop closed palm)
    spread_im = _dist2d(lm[8], lm[12]) / palm_scale
    spread_mr = _dist2d(lm[12], lm[16]) / palm_scale
    spread_rp = _dist2d(lm[16], lm[20]) / palm_scale

    # 16-17: Index pointing horizontal vector (for "Go")
    idx_dx = abs(float(lm[8][0] - lm[5][0])) / palm_scale
    idx_abs_dy = abs(float(lm[8][1] - lm[5][1])) / palm_scale

    # 18: Fist compactness
    fist_compact = float(np.mean(compact_dists))

    # 19: Palm upright orientation (positive when knuckles above wrist)
    palm_upright = (wrist[1] - lm[9][1]) / palm_scale

    return np.array(
        [
            exts[0],
            exts[1],
            exts[2],
            exts[3],
            ups[0],
            ups[1],
            ups[2],
            ups[3],
            thumb_ext,
            thumb_up,
            thumb_above_idx,
            pinch_dist,
            spread_im,
            spread_mr,
            spread_rp,
            idx_dx,
            idx_abs_dy,
            fist_compact,
            palm_upright,
        ],
        dtype=np.float64,
    )


def _generate_synthetic_dataset(samples_per_class: int = 260, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate biomechanically calibrated training distributions for static signs."""
    rng = np.random.default_rng(seed)

    # Feature order matches FEATURE_NAMES (19 features):
    # [idx_ext, mid_ext, rng_ext, pnk_ext,
    #  idx_up, mid_up, rng_up, pnk_up,
    #  thumb_ext, thumb_up, thumb_above_idx, pinch_dist,
    #  spread_im, spread_mr, spread_rp,
    #  idx_dx, idx_abs_dy, fist_compact, palm_upright]
    profiles: Dict[str, Tuple[List[float], List[float]]] = {
        "Hello": (
            [1.45, 1.48, 1.44, 1.38,  0.75, 0.82, 0.76, 0.62,  1.28, 0.35, 0.05, 1.15,  0.52, 0.50, 0.54,  0.30, 0.92, 0.98, 0.95],
            [0.14, 0.14, 0.14, 0.14,  0.18, 0.18, 0.18, 0.18,  0.22, 0.22, 0.22, 0.25,  0.11, 0.11, 0.12,  0.18, 0.20, 0.15, 0.18],
        ),
        "Stop": (
            [1.46, 1.50, 1.45, 1.38,  0.80, 0.86, 0.80, 0.68,  0.86, 0.32, -0.05, 0.85, 0.22, 0.21, 0.23,  0.12, 0.96, 0.96, 0.98],
            [0.13, 0.13, 0.13, 0.13,  0.16, 0.16, 0.16, 0.16,  0.22, 0.20, 0.20, 0.24,  0.05, 0.05, 0.05,  0.12, 0.18, 0.14, 0.16],
        ),
        "Thank You": (
            [0.82, 0.78, 0.75, 0.72, -0.10, -0.15, -0.15, -0.12, 1.28, 0.78, 0.62, 0.95, 0.26, 0.24, 0.24,  0.28, 0.24, 0.34, 0.15],
            [0.15, 0.15, 0.14, 0.14,  0.22, 0.20, 0.20, 0.18,  0.22, 0.18, 0.18, 0.24,  0.10, 0.10, 0.10,  0.18, 0.18, 0.12, 0.35],
        ),
        "Yes": (
            [1.06, 1.48, 1.44, 1.36,  0.12, 0.82, 0.76, 0.64,  0.96, 0.36, 0.12, 0.18,  0.65, 0.36, 0.36,  0.35, 0.45, 0.82, 0.90],
            [0.16, 0.14, 0.14, 0.14,  0.24, 0.18, 0.18, 0.18,  0.22, 0.20, 0.20, 0.09,  0.18, 0.14, 0.14,  0.18, 0.22, 0.14, 0.18],
        ),
        "Go": (
            [1.46, 0.82, 0.78, 0.74,  0.02, -0.05, -0.05, -0.04, 0.85, 0.18, 0.10, 1.05, 0.88, 0.24, 0.24,  1.05, 0.18, 0.48, 0.15],
            [0.14, 0.15, 0.14, 0.14,  0.20, 0.20, 0.18, 0.18,  0.24, 0.22, 0.22, 0.25,  0.20, 0.10, 0.10,  0.20, 0.14, 0.14, 0.35],
        ),
        "Peace": (
            [1.48, 1.50, 0.78, 0.74,  0.80, 0.84, -0.25, -0.22, 0.65, 0.12, -0.20, 0.98, 0.48, 0.92, 0.24,  0.25, 0.92, 0.64, 0.92],
            [0.14, 0.14, 0.14, 0.14,  0.18, 0.18, 0.20, 0.18,  0.20, 0.20, 0.22, 0.24,  0.16, 0.20, 0.10,  0.16, 0.18, 0.12, 0.18],
        ),
        "Help": (
            [0.78, 0.76, 0.74, 0.72, -0.26, -0.28, -0.26, -0.22, 0.62, 0.05, -0.18, 0.44, 0.24, 0.22, 0.22,  0.20, 0.22, 0.32, 0.78],
            [0.14, 0.14, 0.14, 0.14,  0.20, 0.20, 0.18, 0.18,  0.18, 0.18, 0.18, 0.16,  0.09, 0.09, 0.09,  0.14, 0.15, 0.12, 0.24],
        ),
    }

    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for label, (means, stds) in profiles.items():
        mean_arr = np.array(means, dtype=np.float64)
        std_arr = np.array(stds, dtype=np.float64)
        samples = rng.normal(loc=mean_arr, scale=std_arr, size=(samples_per_class, len(means)))
        # Keep non-negative distance features realistic
        for idx in (0, 1, 2, 3, 8, 11, 12, 13, 14, 15, 16, 17):
            samples[:, idx] = np.clip(samples[:, idx], 0.02, 2.5)
        X_list.append(samples)
        y_list.extend([label] * samples_per_class)

    return np.vstack(X_list), np.array(y_list)


class SignLanguageAIModel:
    """Trained ML classifier + geometric/temporal specialist for all 9 poster signs."""

    def __init__(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.classifier: Optional[RandomForestClassifier] = None
        self.model_accuracy: float = 0.0
        self.custom_counts: Dict[str, int] = {label: 0 for label in SIGN_LABELS}
        self.last_inference_ms: float = 0.0

        # Temporal wave history for "Good Bye"
        self._wave_history: deque[Tuple[float, float, float]] = deque(maxlen=26)
        self._goodbye_hold_until: float = 0.0

        # Prediction smoothing buffer to avoid single-frame jitter
        self._recent_preds: deque[str] = deque(maxlen=7)

        self._load_or_train()

    def _load_custom_samples(self) -> Tuple[List[List[float]], List[str]]:
        if not CUSTOM_SAMPLES_PATH.exists():
            return [], []
        try:
            payload = json.loads(CUSTOM_SAMPLES_PATH.read_text(encoding="utf-8"))
            X_custom = payload.get("features", [])
            y_custom = payload.get("labels", [])
            self.custom_counts = {label: 0 for label in SIGN_LABELS}
            for lbl in y_custom:
                if lbl in self.custom_counts:
                    self.custom_counts[lbl] += 1
            return X_custom, y_custom
        except Exception:
            return [], []

    def _save_custom_samples(self, features: List[List[float]], labels: List[str]) -> None:
        payload = {"features": features, "labels": labels}
        CUSTOM_SAMPLES_PATH.write_text(json.dumps(payload), encoding="utf-8")

    def add_custom_sample(self, label: str, landmarks: np.ndarray) -> Dict:
        """Record a live hand landmark sample for interactive retraining."""
        if label not in SIGN_LABELS:
            raise ValueError(f"Unknown sign label: {label}")
        feat = extract_hand_features(landmarks).tolist()
        X_custom, y_custom = self._load_custom_samples()
        X_custom.append(feat)
        y_custom.append(label)
        self._save_custom_samples(X_custom, y_custom)
        self.custom_counts[label] = self.custom_counts.get(label, 0) + 1
        return {
            "label": label,
            "total_for_label": self.custom_counts[label],
            "total_custom_samples": len(y_custom),
        }

    def clear_custom_samples(self) -> None:
        if CUSTOM_SAMPLES_PATH.exists():
            CUSTOM_SAMPLES_PATH.unlink()
        self.custom_counts = {label: 0 for label in SIGN_LABELS}
        self.train_and_save()

    def train_and_save(self) -> Dict:
        """Train RandomForestClassifier on calibrated dataset + user samples and save to disk."""
        X_syn, y_syn = _generate_synthetic_dataset(samples_per_class=260)
        X_custom, y_custom = self._load_custom_samples()

        if X_custom:
            # Give user-recorded samples extra weight (5x) so personal hand calibration shines
            X_c = np.repeat(np.asarray(X_custom, dtype=np.float64), 5, axis=0)
            y_c = np.repeat(np.asarray(y_custom), 5, axis=0)
            X_all = np.vstack([X_syn, X_c])
            y_all = np.concatenate([y_syn, y_c])
        else:
            X_all = X_syn
            y_all = y_syn

        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        scores = cross_val_score(clf, X_all, y_all, cv=4)
        clf.fit(X_all, y_all)

        self.classifier = clf
        self.model_accuracy = float(np.mean(scores))

        with MODEL_PATH.open("wb") as f:
            pickle.dump(
                {
                    "classifier": clf,
                    "accuracy": self.model_accuracy,
                    "feature_names": FEATURE_NAMES,
                },
                f,
            )

        return {
            "accuracy": round(self.model_accuracy * 100, 1),
            "samples_trained": int(len(y_all)),
            "custom_samples": len(y_custom),
            "model_path": str(MODEL_PATH),
        }

    def _load_or_train(self) -> None:
        self._load_custom_samples()
        if MODEL_PATH.exists():
            try:
                with MODEL_PATH.open("rb") as f:
                    bundle = pickle.load(f)
                if bundle.get("feature_names") == FEATURE_NAMES and "classifier" in bundle:
                    self.classifier = bundle["classifier"]
                    self.model_accuracy = float(bundle.get("accuracy", 0.98))
                    return
            except Exception:
                pass
        self.train_and_save()

    def reset_temporal(self) -> None:
        self._wave_history.clear()
        self._recent_preds.clear()

    def _detect_please_two_hands(self, hands_landmarks: List[np.ndarray]) -> Optional[Tuple[str, float]]:
        """Detect 'Please' — two hands pressed together upright (namaste/prayer pose)."""
        if len(hands_landmarks) < 2:
            return None

        h1, h2 = hands_landmarks[0], hands_landmarks[1]
        palm_dist = _dist2d(h1[9], h2[9])
        tip_dist = _dist2d(h1[12], h2[12])
        wrist_dist = _dist2d(h1[0], h2[0])

        # Check both hands are roughly upright (middle tip above wrist)
        h1_upright = (h1[0][1] - h1[12][1]) > 0.05
        h2_upright = (h2[0][1] - h2[12][1]) > 0.05

        if h1_upright and h2_upright and (palm_dist < 0.24 or (tip_dist < 0.19 and wrist_dist < 0.26)):
            closeness = max(0.0, 1.0 - (min(palm_dist, tip_dist) / 0.28))
            conf = min(0.99, 0.84 + closeness * 0.15)
            return ("Please", conf)
        return None

    def _detect_goodbye_wave(self, lm: np.ndarray, features: np.ndarray) -> Optional[Tuple[str, float]]:
        """Detect 'Good Bye' — open hand waving side to side."""
        now = time.time()

        # Check if hand is open (at least 3 of 4 fingers clearly extended upright)
        ext_count = sum(1 for i in range(4) if features[i] > 1.18 and features[4 + i] > 0.30)
        if ext_count < 3:
            # Clear wave history if hand closes
            self._wave_history.clear()
            return None

        wrist_x = float(lm[0][0])
        tip_x = float(lm[12][0])
        self._wave_history.append((now, wrist_x, tip_x))

        # Keep only samples within last 0.95 seconds
        while self._wave_history and (now - self._wave_history[0][0]) > 0.95:
            self._wave_history.popleft()

        if len(self._wave_history) >= 7:
            xs = [pt[2] for pt in self._wave_history]
            diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            total_travel = sum(abs(d) for d in diffs)
            span = max(xs) - min(xs)

            # Count meaningful horizontal direction reversals (oscillation)
            significant_dirs = []
            for d in diffs:
                if abs(d) > 0.011:
                    significant_dirs.append(1 if d > 0 else -1)

            reversals = 0
            for i in range(1, len(significant_dirs)):
                if significant_dirs[i] != significant_dirs[i - 1]:
                    reversals += 1

            if (reversals >= 1 and total_travel >= 0.14 and span >= 0.06) or (
                reversals >= 2 and total_travel >= 0.10
            ):
                self._goodbye_hold_until = now + 0.85
                conf = min(0.98, 0.84 + min(0.14, total_travel * 0.35))
                return ("Good Bye", conf)

        if now < self._goodbye_hold_until:
            return ("Good Bye", 0.91)

        return None

    def _geometric_boost(self, features: np.ndarray, probs: Dict[str, float]) -> Dict[str, float]:
        """Gentle biomechanical sanity refinement on top of RandomForest probabilities."""
        idx_ext, mid_ext, rng_ext, pnk_ext = features[0:4]
        idx_up, mid_up, rng_up, pnk_up = features[4:8]
        thumb_ext, thumb_up, thumb_above_idx, pinch_dist = features[8:12]
        spread_im, spread_mr, spread_rp = features[12:15]
        idx_dx, idx_abs_dy, fist_compact, _ = features[15:19]

        avg_spread = (spread_im + spread_mr + spread_rp) / 3.0
        refined = dict(probs)

        # Strong OK-sign pinch with other 3 fingers open -> Yes
        if pinch_dist < 0.35 and mid_ext > 1.22 and rng_ext > 1.18 and pnk_ext > 1.12:
            refined["Yes"] = refined.get("Yes", 0.0) + 0.38

        # All 4 fingers clearly open & upright -> cleanly separate Hello (spread) vs Stop (together)
        if idx_ext > 1.22 and mid_ext > 1.24 and rng_ext > 1.20 and pnk_ext > 1.15 and mid_up > 0.35:
            if avg_spread >= 0.34:
                refined["Hello"] = refined.get("Hello", 0.0) + 0.32
            else:
                refined["Stop"] = refined.get("Stop", 0.0) + 0.32

        # Index & middle up, ring & pinky curled -> Peace
        if idx_ext > 1.22 and mid_ext > 1.22 and rng_ext < 1.10 and pnk_ext < 1.08 and mid_up > 0.35:
            refined["Peace"] = refined.get("Peace", 0.0) + 0.32

        # Index pointing horizontally, other 3 curled -> Go
        if idx_ext > 1.20 and idx_dx > 0.65 and idx_abs_dy < 0.55 and mid_ext < 1.12 and rng_ext < 1.10:
            refined["Go"] = refined.get("Go", 0.0) + 0.36

        # All 4 fingers curled -> distinguish Thank You (thumb high) vs Help (closed fist)
        if idx_ext < 1.10 and mid_ext < 1.08 and rng_ext < 1.06 and pnk_ext < 1.05 and fist_compact < 0.58:
            if thumb_up > 0.42 and thumb_above_idx > 0.25:
                refined["Thank You"] = refined.get("Thank You", 0.0) + 0.36
            elif thumb_up < 0.32:
                refined["Help"] = refined.get("Help", 0.0) + 0.32

        # Renormalize probabilities
        total = sum(max(0.0, v) for v in refined.values()) + 1e-9
        return {k: round(max(0.0, v) / total, 4) for k, v in refined.items()}

    def predict(self, hands_landmarks: List[np.ndarray]) -> Dict:
        """Run AI inference on 1 or 2 detected hands."""
        t0 = time.perf_counter()

        if not hands_landmarks:
            self._recent_preds.clear()
            self.last_inference_ms = 0.0
            return {
                "sign": None,
                "confidence": 0.0,
                "raw_sign": None,
                "probabilities": {},
                "inference_ms": 0.0,
                "stable": False,
            }

        # Priority 1: Two-hand "Please" gesture
        please_res = self._detect_please_two_hands(hands_landmarks)
        if please_res is not None:
            sign, conf = please_res
            probs = {label: (conf if label == sign else round((1.0 - conf) / 8, 4)) for label in SIGN_LABELS}
            return self._finalize(sign, conf, probs, t0)

        # Extract biomechanical features for primary hand
        primary = hands_landmarks[0]
        features = extract_hand_features(primary)

        # Priority 2: Temporal waving "Good Bye" gesture
        goodbye_res = self._detect_goodbye_wave(primary, features)
        if goodbye_res is not None:
            sign, conf = goodbye_res
            probs = {label: (conf if label == sign else round((1.0 - conf) / 8, 4)) for label in SIGN_LABELS}
            return self._finalize(sign, conf, probs, t0)

        # Priority 3: Trained RandomForestClassifier + geometric refinement
        assert self.classifier is not None
        raw_probs = self.classifier.predict_proba(features.reshape(1, -1))[0]
        classes = list(self.classifier.classes_)
        prob_map = {str(cls): float(p) for cls, p in zip(classes, raw_probs)}
        for lbl in SIGN_LABELS:
            prob_map.setdefault(lbl, 0.0)

        refined = self._geometric_boost(features, prob_map)
        best_sign = max(refined, key=lambda k: refined[k])
        best_conf = float(refined[best_sign])

        return self._finalize(best_sign, best_conf, refined, t0)

    def _finalize(self, raw_sign: str, conf: float, probs: Dict[str, float], t0: float) -> Dict:
        self.last_inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if conf >= 0.44:
            self._recent_preds.append(raw_sign)
        else:
            self._recent_preds.clear()

        stable_sign = None
        stable = False
        if len(self._recent_preds) >= 4:
            most_common, count = Counter(self._recent_preds).most_common(1)[0]
            if count >= 4:
                stable_sign = most_common
                stable = True

        # Return top-3 probabilities sorted descending
        top_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True)[:4])

        return {
            "sign": stable_sign or raw_sign,
            "confidence": round(conf, 3),
            "raw_sign": raw_sign,
            "probabilities": top_probs,
            "inference_ms": self.last_inference_ms,
            "stable": stable,
        }
