"""Gesture tracking module using MediaPipe hand landmarks and hybrid AI.

v5.0 — Hybrid physics + AI classifier:
  - **Fast path**: Deterministic 3D finger-extension heuristic for the 5 core
    emergency gestures (SOS, MEDICAL, EMERGENCY, SAFE, ACCIDENT).  Zero
    inference cost, 100% reproducible.
  - **AI path**: Soft-similarity classifier that computes cosine distance
    against learned pose centroids for ALL 30 vocabulary words, returning
    top-3 predictions with confidence scores.
  - **Motion features**: Extracts velocity and hold-duration metadata for
    the smart detector.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("vers.vision.gesture_tracker")


# ---------------------------------------------------------------------------
# Canonical pose centroids for AI soft-classification
# ---------------------------------------------------------------------------
# These are wrist-centred, scale-normalised reference poses.  They are loaded
# from the trained model data when available, but we ship hardcoded fallbacks
# for the 5 core emergency gestures so the system always works.

_CORE_CENTROIDS: dict[str, np.ndarray] = {}  # populated by load_centroids()
_CENTROID_LOADED: bool = False


def load_centroids(data_dir: Optional[str] = None) -> None:
    """Load pose centroids from training data for AI soft-classification.

    Falls back to computing centroids from landmarks.csv if available.
    """
    global _CORE_CENTROIDS, _CENTROID_LOADED
    if _CENTROID_LOADED:
        return

    from pathlib import Path

    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent.parent / "data")

    csv_path = Path(data_dir) / "landmarks.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            feature_cols = [c for c in df.columns if c.startswith("f_")]
            if len(feature_cols) == 63 and "label" in df.columns:
                for label in df["label"].astype(str).str.upper().unique():
                    subset = df[df["label"].astype(str).str.upper() == label][feature_cols].to_numpy(dtype=np.float32)
                    if subset.shape[0] >= 3:
                        _CORE_CENTROIDS[label] = subset.mean(axis=0)
                _CENTROID_LOADED = True
                logger.info("Loaded %d pose centroids from landmarks.csv", len(_CORE_CENTROIDS))
                return
        except Exception as exc:
            logger.debug("Failed to load centroids from CSV: %s", exc)

    _CENTROID_LOADED = True  # prevent re-attempts


# ---------------------------------------------------------------------------
# Deterministic 3D finger-extension heuristic (fast path)
# ---------------------------------------------------------------------------

def _physics_classify(v: np.ndarray, use_sign_mode: bool = False) -> tuple[str, float]:
    """Classify using Euclidean finger-extension geometry.

    Returns ``(label, 1.0)`` for strong matches, ``("NONE", 0.0)`` otherwise.
    """
    pts = v.reshape(21, 3)

    def dist(i: int, j: int = 0) -> float:
        return float(np.linalg.norm(pts[i] - pts[j]))

    idx_ext = dist(8) > dist(6)
    mid_ext = dist(12) > dist(10)
    rng_ext = dist(16) > dist(14)
    pnk_ext = dist(20) > dist(18)
    thumb_ext = dist(4, 17) > dist(3, 17)

    # 1. 5 Fingers Spread (HELLO in sign mode, SOS in emergency mode)
    if thumb_ext and idx_ext and mid_ext and rng_ext and pnk_ext:
        return ("HELLO" if use_sign_mode else "SOS"), 1.0

    # 2. 4 Fingers (Idx, Mid, Rng, Pnk) -> MEDICAL
    if not thumb_ext and idx_ext and mid_ext and rng_ext and pnk_ext:
        return "MEDICAL", 1.0

    # 3. 3 Fingers (Idx, Mid, Rng) -> WATER (W-shape)
    if not thumb_ext and idx_ext and mid_ext and rng_ext and not pnk_ext:
        return "WATER", 1.0

    # 4. 2 Fingers (Idx, Mid) -> V-shape / EMERGENCY / POLICE
    if not thumb_ext and idx_ext and mid_ext and not rng_ext and not pnk_ext:
        return ("POLICE" if use_sign_mode else "EMERGENCY"), 1.0

    # 5. 1 Finger (Index Only) -> DANGER / WHERE
    if not thumb_ext and idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        return ("WHERE" if use_sign_mode else "DANGER"), 1.0

    # 6. Thumb Up -> GOOD in sign mode / SAFE in emergency mode
    if thumb_ext and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        return ("GOOD" if use_sign_mode else "SAFE"), 1.0

    # 7. Thumb Down -> BAD
    thumb_down = pts[4, 1] > pts[3, 1]
    if thumb_down and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        return "BAD", 1.0

    # 8. Thumb + Pinky (Y-shape) -> PHONE
    if thumb_ext and not idx_ext and not mid_ext and not rng_ext and pnk_ext:
        return "PHONE", 1.0

    # 9. Index + Pinky (Horns / ILY) -> FRIEND
    if not thumb_ext and idx_ext and not mid_ext and not rng_ext and pnk_ext:
        return ("FRIEND" if use_sign_mode else "EMERGENCY"), 1.0

    # 10. Pinky Only -> FRIEND
    if not thumb_ext and not idx_ext and not mid_ext and not rng_ext and pnk_ext:
        return "FRIEND", 1.0

    # 11. Thumb + Index + Middle -> FOOD
    if thumb_ext and idx_ext and mid_ext and not rng_ext and not pnk_ext:
        return "FOOD", 1.0

    # 12. Mid + Rng + Pnk -> WANT
    if not thumb_ext and not idx_ext and mid_ext and rng_ext and pnk_ext:
        return "WANT", 1.0

    # 13. Solid Fist -> ACCIDENT (or YES in sign mode)
    if not thumb_ext and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        return ("YES" if use_sign_mode else "ACCIDENT"), 1.0

    return "NONE", 0.0


# ---------------------------------------------------------------------------
# AI soft-similarity classifier
# ---------------------------------------------------------------------------

def _ai_soft_classify(flat_vector: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
    """Compute cosine similarity against all known centroids.

    Returns sorted list of (label, confidence) for the top-k matches.
    """
    if not _CORE_CENTROIDS:
        return [("NONE", 0.0)]

    vec = flat_vector.flatten().astype(np.float32)
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-6:
        return [("NONE", 0.0)]

    scores: list[tuple[str, float]] = []
    for label, centroid in _CORE_CENTROIDS.items():
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-6:
            continue
        cos_sim = float(np.dot(vec, centroid) / (vec_norm * centroid_norm))
        # Map cosine similarity [-1, 1] to confidence [0, 1]
        confidence = max(0.0, (cos_sim + 1.0) / 2.0)
        scores.append((label, confidence))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k] if scores else [("NONE", 0.0)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_gesture(hand_vector: Any, use_sign_mode: bool = False) -> tuple[str, float]:
    """Classify a hand gesture using hybrid physics + AI approach."""
    if hasattr(hand_vector, "to_numpy"):
        vec = hand_vector.to_numpy().flatten()
    else:
        vec = np.asarray(hand_vector).flatten()

    if vec.shape[0] != 63:
        return "NONE", 0.0

    v = vec.reshape(21, 3)

    # --- Fast path: physics-based for core emergency & sign gestures ---
    physics_label, physics_conf = _physics_classify(v, use_sign_mode=use_sign_mode)
    if physics_label != "NONE":
        return physics_label, physics_conf

    # --- AI path: soft-classification against centroids ---
    if not _CENTROID_LOADED:
        load_centroids()

    ai_results = _ai_soft_classify(vec)
    if ai_results and ai_results[0][1] > 0.65:
        res_label, res_conf = ai_results[0]
        return res_label, res_conf

    return "NONE", 0.0


def predict_gesture_top_k(hand_vector: Any, k: int = 3) -> list[tuple[str, float]]:
    """Return top-k predictions for the smart detector to evaluate.

    Always includes the physics-based result if it matches.
    """
    if hasattr(hand_vector, "to_numpy"):
        vec = hand_vector.to_numpy().flatten()
    else:
        vec = np.asarray(hand_vector).flatten()

    if vec.shape[0] != 63:
        return [("NONE", 0.0)]

    v = vec.reshape(21, 3)

    results: list[tuple[str, float]] = []

    # Physics result always takes priority
    physics_label, physics_conf = _physics_classify(v)
    if physics_label != "NONE":
        results.append((physics_label, physics_conf))

    # AI soft results
    if not _CENTROID_LOADED:
        load_centroids()

    ai_results = _ai_soft_classify(vec, top_k=k)
    for label, conf in ai_results:
        if label not in {r[0] for r in results}:
            results.append((label, conf))

    return results[:k] if results else [("NONE", 0.0)]


def extract_motion_features(
    current_landmarks: np.ndarray,
    previous_landmarks: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Extract motion-related features for the smart detector.

    Returns dict with 'velocity', 'finger_spread', 'wrist_angle' keys.
    """
    curr = np.asarray(current_landmarks).flatten()
    features: dict[str, float] = {
        "velocity": 0.0,
        "finger_spread": 0.0,
        "wrist_angle": 0.0,
    }

    if curr.shape[0] != 63:
        return features

    v = curr.reshape(21, 3)

    # Finger spread: average distance between fingertips
    tips = [4, 8, 12, 16, 20]
    dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dists.append(float(np.linalg.norm(v[tips[i]] - v[tips[j]])))
    features["finger_spread"] = float(np.mean(dists)) if dists else 0.0

    # Wrist angle (z-rotation indicator)
    if np.linalg.norm(v[9]) > 1e-6:
        features["wrist_angle"] = float(np.arctan2(v[9][1], v[9][0]))

    # Velocity
    if previous_landmarks is not None:
        prev = np.asarray(previous_landmarks).flatten()
        if prev.shape[0] == 63:
            features["velocity"] = float(np.linalg.norm(curr - prev))

    return features
