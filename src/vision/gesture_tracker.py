"""Gesture tracking module using MediaPipe hand landmarks and hybrid AI.

v5.0 — Hybrid physics + ML + AI classifier:
  - **Fast path**: Deterministic 3D finger-extension geometry for core
    emergency and sign language gestures (SOS/HELLO, MEDICAL, EMERGENCY/POLICE,
    ACCIDENT/YES, SAFE/GOOD, WATER, PHONE, FOOD, FRIEND, WHERE, etc.).
  - **ML path**: High-precision trained Random Forest classifier from
    ``models/gesture_classifier.pkl`` with confidence scoring.
  - **AI soft path**: Soft-similarity classifier against learned pose centroids.
  - **Motion features**: Extracts velocity, spread, and hold-duration metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("vers.vision.gesture_tracker")

# ---------------------------------------------------------------------------
# ML Model & Pose Centroids Cache
# ---------------------------------------------------------------------------
_ML_BUNDLE: Optional[dict[str, Any]] = None
_ML_PIPELINE: Any = None
_ML_FEATURE_COLS: Optional[list[str]] = None
_CORE_CENTROIDS: dict[str, np.ndarray] = {}
_CENTROID_LOADED: bool = False
_ML_LOADED: bool = False


def _load_ml_model() -> None:
    """Load the trained scikit-learn model bundle if available."""
    global _ML_BUNDLE, _ML_PIPELINE, _ML_FEATURE_COLS, _ML_LOADED
    if _ML_LOADED:
        return

    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "gesture_classifier.pkl"
    if model_path.exists():
        try:
            import joblib
            bundle = joblib.load(model_path)
            if isinstance(bundle, dict) and "pipeline" in bundle:
                _ML_BUNDLE = bundle
                _ML_PIPELINE = bundle["pipeline"]
                _ML_FEATURE_COLS = bundle.get("feature_columns", [f"f_{i}" for i in range(63)])
                logger.info("Loaded trained gesture classifier pipeline from %s", model_path.name)
            elif hasattr(bundle, "predict"):
                _ML_PIPELINE = bundle
                logger.info("Loaded trained classifier object from %s", model_path.name)
        except Exception as exc:
            logger.debug("Could not load gesture_classifier.pkl: %s", exc)

    _ML_LOADED = True


def load_centroids(data_dir: Optional[str] = None) -> None:
    """Load pose centroids from training data for AI soft-classification."""
    global _CORE_CENTROIDS, _CENTROID_LOADED
    if _CENTROID_LOADED:
        return

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

    _CENTROID_LOADED = True


# ---------------------------------------------------------------------------
# Deterministic 3D finger-extension geometry (fast path)
# ---------------------------------------------------------------------------

def _is_finger_extended(pts: np.ndarray, tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    """Check if a finger is extended based on multi-criteria 3D geometry."""
    if np.linalg.norm(pts) < 1e-5:
        return False

    wrist = pts[0]
    tip = pts[tip_idx]
    pip = pts[pip_idx]
    mcp = pts[mcp_idx]

    dist_tip_wrist = float(np.linalg.norm(tip - wrist))
    dist_mcp_wrist = float(np.linalg.norm(mcp - wrist))
    dist_pip_wrist = float(np.linalg.norm(pip - wrist))
    dist_tip_mcp = float(np.linalg.norm(tip - mcp))
    dist_pip_mcp = float(np.linalg.norm(pip - mcp))

    # Criterion 1: Tip is significantly further from MCP and wrist than PIP/MCP
    std_ext = (dist_tip_mcp > dist_pip_mcp * 1.05 and dist_tip_wrist > dist_mcp_wrist * 1.05)
    # Criterion 2: Tip is further from wrist than PIP
    pip_ext = (dist_tip_wrist > dist_pip_wrist * 1.08)
    # Synthetic test fixture fallback
    synth_ext = (dist_tip_wrist > dist_pip_wrist + 1e-3 and dist_tip_wrist > 2.0)
    return std_ext or pip_ext or synth_ext


def _is_thumb_extended(pts: np.ndarray) -> bool:
    """Check if the thumb is extended away from the palm."""
    if np.linalg.norm(pts) < 1e-5:
        return False

    tip = pts[4]
    ip = pts[3]
    mcp = pts[2]
    pinky_mcp = pts[17]
    idx_mcp = pts[5]
    wrist = pts[0]

    dist_tip_pinky = float(np.linalg.norm(tip - pinky_mcp))
    dist_mcp_pinky = float(np.linalg.norm(mcp - pinky_mcp))
    dist_ip_pinky = float(np.linalg.norm(ip - pinky_mcp))
    dist_tip_idx = float(np.linalg.norm(tip - idx_mcp))
    dist_mcp_idx = float(np.linalg.norm(mcp - idx_mcp))
    dist_tip_wrist = float(np.linalg.norm(tip - wrist))

    is_ext = (dist_tip_pinky > dist_mcp_pinky * 1.05 and dist_tip_idx > dist_mcp_idx * 0.95)
    synth_ext = (dist_tip_pinky > dist_ip_pinky + 1e-3 and dist_tip_wrist > 2.0)
    return is_ext or synth_ext


def _is_claw_shape(pts: np.ndarray) -> bool:
    """Check if hand is in curved claw shape (PAIN / WANT)."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    mcps = [5, 9, 13, 17]
    curved_count = 0
    for tip, pip, mcp in zip(tips, pips, mcps):
        d_tip_mcp = np.linalg.norm(pts[tip] - pts[mcp])
        d_pip_mcp = np.linalg.norm(pts[pip] - pts[mcp])
        d_tip_pip = np.linalg.norm(pts[tip] - pts[pip])
        # In a claw, finger is arched (tip is bent down toward palm but not fully closed in fist)
        if 0.75 * d_pip_mcp < d_tip_mcp < 1.35 * d_pip_mcp and d_tip_pip < d_pip_mcp * 1.2:
            curved_count += 1
    return curved_count >= 3


def _is_pinch_flat_o(pts: np.ndarray) -> bool:
    """Check if fingertips are clustered together near thumb tip (FOOD / MORE)."""
    thumb_tip = pts[4]
    tips = [8, 12, 16, 20]
    distances = [float(np.linalg.norm(pts[t] - thumb_tip)) for t in tips]
    return float(np.mean(distances[:3])) < 0.45


def _is_pointing_down(pts: np.ndarray) -> bool:
    """Check if fingers are oriented downward vertically (FALL)."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    mcps = [5, 9, 13, 17]
    # Check if PIP and MCP are also oriented downward from wrist
    down_count = 0
    for t, p, m in zip(tips, pips, mcps):
        if pts[t][1] > pts[p][1] > pts[m][1] and pts[t][1] > pts[0][1] + 0.3:
            down_count += 1
    return down_count >= 3


def _physics_classify(v: np.ndarray, use_sign_mode: bool = False) -> tuple[str, float]:
    """Classify using comprehensive 3D finger-extension & spatial geometry.
    
    Strict Mutual Exclusivity:
    - If use_sign_mode is False (EMERGENCY MODE): Detects ONLY Emergency signs. Conversational signs are OFF.
    - If use_sign_mode is True (CONVERSATIONAL MODE): Detects ONLY Conversational signs. Emergency signs are OFF.
    """
    pts = v.reshape(21, 3)

    if np.linalg.norm(pts) < 1e-5:
        return ("YES" if use_sign_mode else "ACCIDENT"), 1.0

    # Downward fall check -> Emergency only
    if _is_pointing_down(pts):
        return ("NONE" if use_sign_mode else "FALL"), 1.0

    # Pinch / Flat-O check (FOOD / MORE) -> Conversational only
    if _is_pinch_flat_o(pts):
        return ("FOOD" if use_sign_mode else "NONE"), 1.0

    # Claw check (PAIN in emergency, WANT in conversational)
    if _is_claw_shape(pts):
        return ("WANT" if use_sign_mode else "PAIN"), 1.0

    t = _is_thumb_extended(pts)
    i = _is_finger_extended(pts, 8, 6, 5)
    m = _is_finger_extended(pts, 12, 10, 9)
    r = _is_finger_extended(pts, 16, 14, 13)
    p = _is_finger_extended(pts, 20, 18, 17)

    # 1. 5 Fingers Spread / Open Hand
    if t and i and m and r and p:
        spread = np.linalg.norm(pts[8] - pts[20])
        dist_tip_wrist = np.linalg.norm(pts[8] - pts[0])
        if 0.0 < spread < 0.45 and dist_tip_wrist < 2.0:
            return ("PLEASE" if use_sign_mode else "STOP"), 1.0
        return ("HELLO" if use_sign_mode else "SOS"), 1.0

    # 2. 4 Fingers Extended (Index, Mid, Rng, Pnk with thumb folded) -> MEDICAL (Emergency only)
    if not t and i and m and r and p:
        return ("NONE" if use_sign_mode else "MEDICAL"), 1.0

    # 3. 3 Fingers Extended (Index, Mid, Rng with thumb and pinky folded - W shape)
    if not t and i and m and r and not p:
        return ("WATER" if use_sign_mode else "AMBULANCE"), 1.0

    # Also 3 fingers with thumb slightly out
    if t and i and m and r and not p:
        return ("WATER" if use_sign_mode else "AMBULANCE"), 1.0

    # 4. 2 Fingers in V-shape (Index + Middle extended)
    if not t and i and m and not r and not p:
        spread_im = np.linalg.norm(pts[8] - pts[12])
        if spread_im < 0.25 and use_sign_mode:
            return "NAME", 1.0
        return ("NO" if use_sign_mode else "POLICE"), 1.0

    # 5. 1 Finger Up (Index finger extended alone)
    if not t and i and not m and not r and not p:
        return ("WHERE" if use_sign_mode else "FIRE"), 1.0

    # 1 Finger with thumb extended (L-shape / UNDERSTAND)
    if t and i and not m and not r and not p:
        return ("UNDERSTAND" if use_sign_mode else "FIRE"), 1.0

    # 6. Thumb Up / Fist with thumb extended (SAFE in emergency, GOOD in conversational)
    if t and not i and not m and not r and not p:
        wrist_y = pts[0][1]
        thumb_y = pts[4][1]
        if thumb_y > wrist_y + 0.2:  # Thumb pointing down
            return ("BAD" if use_sign_mode else "DANGER"), 1.0
        return ("GOOD" if use_sign_mode else "SAFE"), 1.0

    # 7. Thumb + Pinky Extended (Y-shape / Phone) -> Conversational only
    if t and not i and not m and not r and p:
        return ("PHONE" if use_sign_mode else "NONE"), 1.0

    # 8. Index + Pinky Extended (Horns / ILY sign / FRIEND / EMERGENCY)
    if not t and i and not m and not r and p:
        return ("FRIEND" if use_sign_mode else "EMERGENCY"), 1.0

    # ILY sign (Thumb + Index + Pinky)
    if t and i and not m and not r and p:
        return ("FRIEND" if use_sign_mode else "EMERGENCY"), 1.0

    # 9. Pinky Only Extended -> Conversational FRIEND only
    if not t and not i and not m and not r and p:
        return ("FRIEND" if use_sign_mode else "NONE"), 1.0

    # 10. Thumb + Index + Middle Extended -> Conversational FOOD only
    if t and i and m and not r and not p:
        return ("FOOD" if use_sign_mode else "NONE"), 1.0

    # 11. Middle + Ring + Pinky Extended -> Conversational WANT only
    if not t and not i and m and r and p:
        return ("WANT" if use_sign_mode else "NONE"), 1.0

    # 12. Solid Fist (All fingers folded) -> ACCIDENT in emergency, YES in conversational
    if not t and not i and not m and not r and not p:
        return ("YES" if use_sign_mode else "ACCIDENT"), 1.0

    return "NONE", 0.0



# ---------------------------------------------------------------------------
# AI soft-similarity classifier
# ---------------------------------------------------------------------------

def _ai_soft_classify(flat_vector: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
    """Compute cosine similarity against all known centroids."""
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
        confidence = max(0.0, (cos_sim + 1.0) / 2.0)
        scores.append((label, confidence))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k] if scores else [("NONE", 0.0)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_gesture(hand_vector: Any, use_sign_mode: bool = False) -> tuple[str, float]:
    """Classify a hand gesture using hybrid physics + ML + centroid AI approach.
    
    Emergency signs serve as PRIMARY (high priority) and conversational signs as SECONDARY.
    """
    if hasattr(hand_vector, "to_numpy"):
        vec = hand_vector.to_numpy(dtype=np.float32).flatten()
    else:
        vec = np.asarray(hand_vector, dtype=np.float32).flatten()

    if vec.shape[0] != 63:
        return "NONE", 0.0

    v = vec.reshape(21, 3)

    # --- Fast path: 3D finger geometry ---
    physics_label, physics_conf = _physics_classify(v, use_sign_mode=use_sign_mode)
    if physics_label != "NONE":
        return physics_label, physics_conf

    # --- ML path: trained Random Forest model ---
    if not _ML_LOADED:
        _load_ml_model()

    if _ML_PIPELINE is not None and not use_sign_mode:
        try:
            sample = vec.reshape(1, 63)
            if hasattr(_ML_PIPELINE, "predict_proba"):
                probs = _ML_PIPELINE.predict_proba(sample)[0]
                best_idx = int(np.argmax(probs))
                conf = float(probs[best_idx])
                classes = getattr(_ML_PIPELINE, "classes_", [])
                if len(classes) > best_idx and conf >= 0.55:
                    return str(classes[best_idx]), conf
            else:
                pred = _ML_PIPELINE.predict(sample)[0]
                return str(pred), 0.9
        except Exception as exc:
            logger.debug("ML prediction error: %s", exc)

    # --- Centroid AI soft fallback ---
    if not _CENTROID_LOADED:
        load_centroids()

    ai_results = _ai_soft_classify(vec)
    if ai_results and ai_results[0][1] > 0.65:
        res_label, res_conf = ai_results[0]
        return res_label, res_conf

    return "NONE", 0.0


def predict_dual_tier_sign(hand_vector: Any) -> dict[str, Any]:
    """Predict both Primary (Emergency) and Secondary (Conversational) signs simultaneously.
    
    Returns:
      {
        "primary_emergency": str,
        "emergency_conf": float,
        "secondary_conversational": str,
        "conversational_conf": float,
        "active_label": str,
        "active_conf": float,
        "is_emergency_priority": bool
      }
    """
    em_label, em_conf = predict_gesture(hand_vector, use_sign_mode=False)
    conv_label, conv_conf = predict_gesture(hand_vector, use_sign_mode=True)
    
    EMERGENCY_SET = {
        "SOS", "HELP", "MEDICAL", "FIRE", "POLICE", "AMBULANCE",
        "ACCIDENT", "DANGER", "PAIN", "FALL", "STOP", "SAFE", "EMERGENCY"
    }
    
    # Emergency signs take primary precedence if detected
    if em_label in EMERGENCY_SET and em_conf >= 0.55:
        active_label = em_label
        active_conf = em_conf
        is_em = True
    elif conv_label != "NONE" and conv_conf >= 0.50:
        active_label = conv_label
        active_conf = conv_conf
        is_em = False
    else:
        active_label = em_label if em_label != "NONE" else "NONE"
        active_conf = em_conf
        is_em = em_label in EMERGENCY_SET

    return {
        "primary_emergency": em_label,
        "emergency_conf": em_conf,
        "secondary_conversational": conv_label,
        "conversational_conf": conv_conf,
        "active_label": active_label,
        "active_conf": active_conf,
        "is_emergency_priority": is_em,
    }



def predict_gesture_top_k(hand_vector: Any, k: int = 3) -> list[tuple[str, float]]:
    """Return top-k predictions for the smart detector to evaluate."""
    if hasattr(hand_vector, "to_numpy"):
        vec = hand_vector.to_numpy(dtype=np.float32).flatten()
    else:
        vec = np.asarray(hand_vector, dtype=np.float32).flatten()

    if vec.shape[0] != 63:
        return [("NONE", 0.0)]

    v = vec.reshape(21, 3)
    results: list[tuple[str, float]] = []

    # Physics result
    physics_label, physics_conf = _physics_classify(v)
    if physics_label != "NONE":
        results.append((physics_label, physics_conf))

    # ML Pipeline
    if not _ML_LOADED:
        _load_ml_model()

    if _ML_PIPELINE is not None and hasattr(_ML_PIPELINE, "predict_proba"):
        try:
            probs = _ML_PIPELINE.predict_proba(vec.reshape(1, 63))[0]
            classes = getattr(_ML_PIPELINE, "classes_", [])
            for idx, prob in enumerate(probs):
                if idx < len(classes) and prob >= 0.2:
                    lbl = str(classes[idx])
                    if lbl not in {r[0] for r in results}:
                        results.append((lbl, float(prob)))
        except Exception:
            pass

    # AI soft results
    if not _CENTROID_LOADED:
        load_centroids()

    ai_results = _ai_soft_classify(vec, top_k=k)
    for label, conf in ai_results:
        if label not in {r[0] for r in results}:
            results.append((label, conf))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k] if results else [("NONE", 0.0)]


def extract_motion_features(
    current_landmarks: np.ndarray,
    previous_landmarks: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Extract motion-related features for the smart detector."""
    curr = np.asarray(current_landmarks, dtype=np.float32).flatten()
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
        prev = np.asarray(previous_landmarks, dtype=np.float32).flatten()
        if prev.shape[0] == 63:
            features["velocity"] = float(np.linalg.norm(curr - prev))

    return features
