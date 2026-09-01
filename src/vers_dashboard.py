"""VERS v5.0 DashboardRuntime — Multimodal AI Vision Pipeline.

Integrates:
  - Vision: MediaPipe hand tracking + DeepFace facial emotion recognition
  - Intelligence: Temporal sequence smoothing + Multimodal severity fusion
  - Services: Async TTS voice alerts + Simulated GPS alert dispatch
  - Frontend: React Command Center via WebSocket (Streamlit removed)
"""

from __future__ import annotations

import base64
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

MPLCONFIGDIR = Path(os.environ.get("VERS_MPLCONFIGDIR", "/tmp/vers-mplconfig")).resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "0")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Legacy imports (still needed for model loading and overlay drawing) ---
from src.realtime_vers import (
    ALERT_COOLDOWN_SECONDS,
    DISTRESS_THRESHOLD,
    HAND_CONF_THRESHOLD,
    SMOOTHING_WINDOW,
    calc_distress,
    draw_overlay,
    load_model,
)
from src.utils.alert_utils import ALERT_MAP, log_error, make_alert_payload
from src.utils.data_utils import ensure_project_dirs, extract_hand_vector, open_camera_capture

# --- v2.0 Modular Architecture imports ---
from src.vision.gesture_tracker import predict_gesture
from src.vision.emotion_model import analyze_emotion
from src.intelligence.temporal_smoothing import TemporalSmoother
from src.utils.alert_utils import ALERT_MAP, calculate_fused_severity
from src.services.alert_dispatcher import dispatch as dispatch_alert
from src.services import voice_tts
from src.intelligence.smart_detector import SmartDetector

# --- v4.0 Sign Language imports ---
from src.vision.sequence_buffer import SequenceBuffer
from src.vision.sign_language_model import SignLanguageRecognizer
from src.intelligence.intent_mapper import IntentMapper


# Streamlit rerun removed because UI is now in React
class DashboardRuntime:
    """Background webcam worker with v2.0 multimodal AI pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = None
        self._alerts: deque[dict[str, Any]] = deque(maxlen=15)
        self._status: dict[str, Any] = {
            "running": False,
            "camera_active": False,
            "gesture": "No gesture",
            "confidence": 0.0,
            "distress_score": 0.0,
            "distress_flag": False,
            "dominant_emotion": "neutral",
            "emotion_distress": 0.0,
            "severity_score": 0.0,
            "threat_level": "NONE",
            "location": None,
            "fps": 0.0,
            "error": None,
            "labels": [],
            "last_alert": None,
            "starting": False,
        }
        self._conf_threshold = HAND_CONF_THRESHOLD
        self._distress_threshold = DISTRESS_THRESHOLD
        self._model_bundle: dict[str, Any] | None = None
        self._labels: list[str] = []
        self._sign_mode = False  # v4.0: sign language mode toggle
        # Start TTS daemon once
        voice_tts.start()

    def ensure_model_loaded(self) -> list[str]:
        with self._lock:
            if self._model_bundle is None:
                self._model_bundle, self._labels = load_model()
                self._status["labels"] = list(self._labels)
            return list(self._labels)

    def model_bundle(self) -> dict[str, Any]:
        self.ensure_model_loaded()
        with self._lock:
            if self._model_bundle is None:
                raise RuntimeError("Model bundle is not available.")
            return self._model_bundle

    def configure(self, conf_threshold: float, distress_threshold: float, sign_mode: bool = False) -> None:
        with self._lock:
            self._conf_threshold = conf_threshold
            self._distress_threshold = distress_threshold
            self._sign_mode = sign_mode

    def start(self, conf_threshold: float, distress_threshold: float) -> None:
        self.ensure_model_loaded()
        self.configure(conf_threshold, distress_threshold)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._status["error"] = None
            self._status["starting"] = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="vers-streamlit-runtime", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Force release the camera immediately to break any blocking cap.read()
        with self._lock:
            if self._status.get("camera_active"):
                # We try a 'soft' release by signaling the thread, but if it's lagging,
                # the thread will catch the _stop_event.
                pass 
        
        thread = self._thread
        if thread is not None and thread.is_alive():
            # If we were to call cap.release() here, we might cause a race in the thread.
            # Instead, we rely on the _stop_event being checked before the next cap.read().
            # To make it 'buttery smooth', we'll ensure the thread doesn't sleep if it's stopping.
            thread.join(timeout=0.5)
        with self._lock:
            self._thread = None
            self._status["running"] = False
            self._status["camera_active"] = False
            self._status["starting"] = False
            self._status["gesture"] = "No gesture"
            self._status["confidence"] = 0.0
            self._status["distress_score"] = 0.0
            self._status["distress_flag"] = False
            self._status["fps"] = 0.0
            self._frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return {
                **self._status,
                "frame": frame,
                "alerts": list(self._alerts),
            }

    def clear_data(self) -> None:
        with self._lock:
            self._alerts.clear()
            self._status["last_alert"] = None
            self._status["gesture"] = "No gesture"
            self._status["confidence"] = 0.0
            self._status["distress_score"] = 0.0
            self._status["distress_flag"] = False
            self._status["fps"] = 0.0

    def _append_alert(self, payload: dict[str, Any]) -> None:
        self._alerts.appendleft(payload)
        self._status["last_alert"] = payload

    def add_alert(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if not self._alerts or self._alerts[0] != payload:
                self._append_alert(payload)

    def _init_ai_pipeline(self) -> None:
        if hasattr(self, "_pipeline_initialized") and self._pipeline_initialized:
            return
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smoother = TemporalSmoother(window_size=7, min_votes=3)
        self._smart_detector = SmartDetector()
        self._sign_buffer = SequenceBuffer(window_size=30)
        self._sign_recognizer = SignLanguageRecognizer()
        self._intent_mapper = IntentMapper(memory_size=5)
        self._cached_emotion = {
            "dominant_emotion": "neutral",
            "emotion_scores": {},
            "distress_contribution": 0.0,
        }
        self._frame_counter = 0
        self._last_tick = time.perf_counter()
        self._pipeline_initialized = True

    def process_frame(self, frame: np.ndarray, use_sign_mode: bool | None = None, is_flipped: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        """Process a single video frame through the full multimodal AI pipeline."""
        self.ensure_model_loaded()
        self._init_ai_pipeline()

        if use_sign_mode is None:
            with self._lock:
                use_sign_mode = self._sign_mode

        if not is_flipped:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_counter += 1
        frame_counter = self._frame_counter

        # --- 1. HAND GESTURE TRACKING ---
        hand_results = self._hands.process(rgb)
        hand_vector = extract_hand_vector(hand_results)

        raw_label, raw_conf = "NONE", 0.0
        if hand_vector is not None:
            raw_label, raw_conf = predict_gesture(hand_vector, use_sign_mode=use_sign_mode)

            if use_sign_mode:
                # CONVERSATIONAL MODE: ONLY Conversational Signs allowed. Emergency is OFF.
                CONVERSATIONAL_WHITELIST = {
                    "HELLO", "THANK_YOU", "PLEASE", "YES", "NO", "WATER", "FOOD", "WANT", 
                    "MORE", "FRIEND", "FAMILY", "NAME", "GOOD", "BAD", "SORRY", "UNDERSTAND", 
                    "PHONE", "WHERE", "FINISHED"
                }
                if raw_label not in CONVERSATIONAL_WHITELIST:
                    raw_label, raw_conf = "NONE", 0.0
            else:
                # EMERGENCY MODE: ONLY Emergency Signs allowed. Conversational is OFF.
                EMERGENCY_WHITELIST = {
                    "HELP", "SOS", "MEDICAL", "FIRE", "POLICE", "AMBULANCE",
                    "ACCIDENT", "DANGER", "PAIN", "FALL", "STOP", "SAFE", "EMERGENCY"
                }
                if raw_label not in EMERGENCY_WHITELIST:
                    raw_label, raw_conf = "NONE", 0.0

            self._sign_buffer.push(hand_vector)

        self._smoother.push(raw_label, raw_conf)
        smooth_label, smooth_conf = self._smoother.smoothed()

        # --- 2. SIGN LANGUAGE RECOGNITION ---
        sign_word = "NONE"
        sign_conf = 0.0
        if use_sign_mode:
            CONVERSATIONAL_WHITELIST = {
                "HELLO", "THANK_YOU", "PLEASE", "YES", "NO", "WATER", "FOOD", "WANT", 
                "MORE", "FRIEND", "FAMILY", "NAME", "GOOD", "BAD", "SORRY", "UNDERSTAND", 
                "PHONE", "WHERE", "FINISHED"
            }
            if self._sign_recognizer.available and self._sign_buffer.ready:
                lstm_word, lstm_conf = self._sign_recognizer.predict(self._sign_buffer.get_tensor())
                if lstm_word in CONVERSATIONAL_WHITELIST and lstm_conf >= 0.55:
                    sign_word, sign_conf = lstm_word, lstm_conf

            if sign_word == "NONE" and smooth_label in CONVERSATIONAL_WHITELIST and smooth_conf >= 0.4:
                sign_word = smooth_label
                sign_conf = smooth_conf

            if sign_word != "NONE":
                self._intent_mapper.push_word(sign_word)

        # Unify active gesture across modes
        if use_sign_mode:
            active_gesture = sign_word if sign_word != "NONE" else smooth_label
            active_confidence = sign_conf if sign_word != "NONE" else smooth_conf
        else:
            active_gesture = smooth_label
            active_confidence = smooth_conf

        # --- 3. EMOTION & FACE ANALYSIS (Every 4 frames for maximum FPS) ---
        if not hasattr(self, "_cached_face_lms"):
            self._cached_face_lms = None

        if frame_counter % 4 == 0 or self._cached_face_lms is None:
            face_results = self._face_mesh.process(rgb)
            if getattr(face_results, "multi_face_landmarks", None):
                self._cached_face_lms = face_results.multi_face_landmarks[0]
            else:
                self._cached_face_lms = None

            emo_result = analyze_emotion(rgb)
            self._cached_emotion["dominant_emotion"] = emo_result["dominant_emotion"]
            self._cached_emotion["distress_contribution"] = emo_result["distress_contribution"]

        face_lms = self._cached_face_lms

        # --- 4. SMART DETECTOR & FUSION ---
        raw_lms = None
        if getattr(hand_results, "multi_hand_landmarks", None):
            raw_lms = np.array([[p.x, p.y, p.z] for p in hand_results.multi_hand_landmarks[0].landmark], dtype=np.float32).flatten()

        smart_result = self._smart_detector.update(
            landmarks=raw_lms,
            gesture_label=active_gesture,
            gesture_confidence=active_confidence,
            distress_score=self._cached_emotion.get("distress_contribution", 0.0)
        )

        base_sev = ALERT_MAP.get(active_gesture, ALERT_MAP["NONE"])["severity"]
        fused_sev_label, fusion_score = calculate_fused_severity(
            confidence=active_confidence,
            distress_score=smart_result.urgency_score,
            base_severity=base_sev
        )

        with self._lock:
            distress_threshold = self._distress_threshold

        distress_flag = smart_result.urgency_score > distress_threshold

        # --- 5. ALERT DISPATCH (Emergency Mode Only) ---
        payload = None
        if not use_sign_mode and smart_result.gesture_label != "NONE" and smart_result.gesture_confidence >= 0.5:
            if smart_result.context_mode == "EMERGENCY" or smart_result.urgency_score > 0.4:
                payload = dispatch_alert(
                    gesture_label=smart_result.gesture_label,
                    gesture_confidence=smart_result.gesture_confidence,
                    dominant_emotion=self._cached_emotion.get("dominant_emotion", "neutral"),
                    emotion_distress=smart_result.urgency_score,
                    severity_score=fusion_score,
                    threat_level=fused_sev_label,
                    distress_flag=distress_flag,
                    enable_tts=True,
                )
                if payload is not None:
                    with self._lock:
                        self._append_alert(payload)

        # --- 6. OVERLAY VISUALS (Zero Lag) ---

        tick = time.perf_counter()
        fps = 1.0 / max(tick - self._last_tick, 1e-6)
        self._last_tick = tick

        overlay = draw_overlay(
            frame,
            hand_results,
            face_lms,
            smart_result,
            sign_word=active_gesture if (use_sign_mode and active_gesture != "NONE") else "",
            fps=fps,
        )
        frame_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        with self._lock:
            self._frame = frame_rgb
            self._status.update(
                {
                    "running": True,
                    "camera_active": True,
                    "gesture": active_gesture if active_gesture != "NONE" else "No gesture",
                    "confidence": active_confidence if active_gesture != "NONE" else 0.0,
                    "distress_score": smart_result.urgency_score,
                    "distress_flag": distress_flag,
                    "dominant_emotion": self._cached_emotion.get("dominant_emotion", "neutral"),
                    "emotion_distress": smart_result.urgency_score,
                    "severity_score": fusion_score,
                    "threat_level": fused_sev_label,
                    "location": payload.get("Location") if payload else self._status.get("location"),
                    "fps": fps,
                    "error": None,
                    "labels": list(self._labels),
                    "last_alert": payload or self._status.get("last_alert"),
                    "sign_word": active_gesture if use_sign_mode else "",
                    "sign_buffer_words": self._intent_mapper.current_words if use_sign_mode else [],
                    "sign_available": self._sign_recognizer.available,
                }
            )

        return frame_rgb, self.snapshot()

    def _run_loop(self) -> None:
        """Background loop for hardware webcam."""
        ensure_project_dirs()
        cap, backend_info = open_camera_capture(max_index=4, warmup_reads=18)
        if cap is None or not cap.isOpened():
            with self._lock:
                self._status["error"] = "Direct Browser Camera is ready to stream."
                self._status["running"] = True
                self._status["camera_active"] = False
            return

        with self._lock:
            self._status["running"] = True
            self._status["camera_active"] = True
            self._status["starting"] = False
            self._status["error"] = None

        consecutive_capture_failures = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_capture_failures += 1
                    if consecutive_capture_failures >= 45:
                        break
                    time.sleep(0.03)
                    continue
                consecutive_capture_failures = 0
                self.process_frame(frame, is_flipped=False)
                time.sleep(0.005)
        except Exception as exc:
            log_error(f"Dashboard runtime loop failure: {exc}")
        finally:
            cap.release()


_RUNTIME = None
def get_runtime() -> DashboardRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = DashboardRuntime()
    return _RUNTIME



