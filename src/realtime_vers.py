"""Real-time VERS demo: gesture recognition, distress scoring, and alerting.

Phase-1 Alignment
-----------------
- System Design: Input -> Preprocessing -> Feature Extraction -> AI Module -> Alert -> Communication
- Methodology #1-#7: Full real-time pipeline from camera feed to structured alert output
- Objectives 1-7: Gesture recognition, distress analysis, accessibility, and inclusive response
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

MPLCONFIGDIR = Path(os.environ.get("VERS_MPLCONFIGDIR", "/tmp/vers-mplconfig")).resolve()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "0")

import cv2
import joblib
import mediapipe as mp
import numpy as np

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

try:
    from rich import print as rprint
except Exception:  # pragma: no cover - optional dependency
    rprint = print

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.alert_utils import ALERT_MAP, log_alert, log_error, make_alert_payload
from src.utils.data_utils import (
    DATA_PATH,
    DISTRESS_HISTORY_PATH,
    MODEL_PATH,
    ensure_project_dirs,
    extract_hand_vector,
    open_camera_capture,
)

from src.vision.gesture_tracker import predict_gesture, extract_motion_features
from src.intelligence.smart_detector import SmartDetector
from src.vision.sequence_buffer import SequenceBuffer
from src.vision.sign_language_model import SignLanguageRecognizer
from src.intelligence.intent_mapper import IntentMapper

VERS_VERSION = "5.0.0"
ALERT_ENDPOINT = "http://localhost:8000/alert"
DISTRESS_THRESHOLD = 0.055
HAND_CONF_THRESHOLD = 0.55
SMOOTHING_WINDOW = 5
ALERT_COOLDOWN_SECONDS = 5
DISTRESS_HISTORY_LIMIT = 200

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
BROW_LEFT, EYE_LEFT = 70, 159
BROW_RIGHT, EYE_RIGHT = 300, 386


# load_model and predict_gesture have been moved to gesture_tracker.py
# and predict_gesture is imported from there.
# We keep a dummy load_model here to not break the main() function structure
# but it now defers to the gesture_tracker's loading.
def load_model() -> tuple[dict[str, Any], list[str]]:
    from src.vision.gesture_tracker import load_centroids, _CORE_CENTROIDS
    load_centroids()
    return {}, list(_CORE_CENTROIDS.keys())



def calc_distress(face_lms: Any, width: int, height: int) -> float:
    """Compute the demo distress heuristic from MediaPipe face landmarks."""
    if face_lms is None:
        return 0.0

    def pt(index: int) -> np.ndarray:
        landmark = face_lms.landmark[index]
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    try:
        mouth_vertical = np.linalg.norm(pt(MOUTH_TOP) - pt(MOUTH_BOTTOM))
        mouth_horizontal = np.linalg.norm(pt(MOUTH_LEFT) - pt(MOUTH_RIGHT))
        brow_left_gap = np.linalg.norm(pt(BROW_LEFT) - pt(EYE_LEFT))
        brow_right_gap = np.linalg.norm(pt(BROW_RIGHT) - pt(EYE_RIGHT))

        mouth_ratio = mouth_vertical / max(mouth_horizontal, 1e-6)
        brow_ratio = (brow_left_gap + brow_right_gap) / (2 * height)
        return float(0.65 * mouth_ratio + 0.35 * brow_ratio)
    except Exception:
        return 0.0


def smooth_prediction(history: deque[tuple[str, float]]) -> tuple[str, float]:
    """Combine recent predictions into a more stable display label."""
    if not history:
        return "NONE", 0.0

    weighted: Counter[str] = Counter()
    confidences: dict[str, list[float]] = {}
    for label, confidence in history:
        weighted[label] += float(confidence)
        confidences.setdefault(label, []).append(float(confidence))

    valid = {label: score for label, score in weighted.items() if label != "NONE"}
    if not valid:
        return "NONE", 0.0

    best_label = max(valid, key=lambda label: valid[label])
    return best_label, float(np.mean(confidences[best_label]))


def append_distress_history(entries: deque[str], distress_score: float, distress_flag: bool) -> None:
    """Persist the most recent distress scores for later analysis."""
    entries.append(f"{datetime.now().isoformat()},{distress_score:.4f},{distress_flag}")
    DISTRESS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DISTRESS_HISTORY_PATH.open("w", encoding="utf-8") as handle:
        handle.write("timestamp,distress_score,distress_flag\n")
        handle.write("\n".join(entries))
        handle.write("\n")


def draw_overlay(
    frame: np.ndarray,
    hand_results: Any,
    face_lms: Any,
    smart_result: Any,
    sign_word: str = "",
    fps: float = 0.0,
) -> np.ndarray:
    """Render the real-time demo overlay with high-visibility background boxes."""
    overlay = frame.copy()
    display_label = getattr(smart_result, "gesture_label", "NONE")
    display_confidence = getattr(smart_result, "gesture_confidence", 0.0)
    distress_score = getattr(smart_result, "urgency_score", 0.0)
    distress_flag = distress_score > DISTRESS_THRESHOLD
    
    is_accepted = (display_label != "NONE" and display_confidence >= HAND_CONF_THRESHOLD)
    is_none = (display_label == "NONE")

    
    # Text and layout constants
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    x_offset = 15
    y_start = 40
    line_height = 35

    def draw_text_with_bg(img, text, pos, color, bg_color=(0, 0, 0), alpha=0.6):
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tx, ty = pos
        # Draw background rect
        rect_start = (tx - 5, ty - th - 5)
        rect_end = (tx + tw + 5, ty + baseline + 5)
        sub_img = img[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
        if sub_img.size > 0:
            black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 1 - alpha, black_rect, alpha, 0)
            img[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]] = res
        cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

    # 1. Gesture Line
    if is_none:
        g_text = "Gesture: None detected"
        g_color = (200, 200, 200)
    elif is_accepted:
        severity = ALERT_MAP.get(str(display_label).upper(), ALERT_MAP["NONE"]).get("severity", "Low")
        g_text = f"Gesture: {display_label} (MATCH)"
        g_color = (0, 255, 0) if severity == "Low" else (0, 165, 255) if severity == "Medium" else (0, 0, 255)
    else:
        g_text = f"Gesture: {display_label} (LOW CONFIDENCE)"
        g_color = (0, 255, 255) # Yellow/Orange

    draw_text_with_bg(overlay, g_text, (x_offset, y_start), g_color)

    # 2. Confidence Line
    # c_color = (0, 255, 0) if is_accepted else (0, 255, 255) if not is_none else (200, 200, 200)
    # draw_text_with_bg(overlay, f"Confidence: {display_confidence:.2f}", (x_offset, y_start + line_height), c_color)

    # 3. Distress Line
    d_color = (0, 0, 255) if distress_flag else (0, 255, 0)
    draw_text_with_bg(overlay, f"Distress: {distress_score:.3f} {'(HIGH)' if distress_flag else '(NORMAL)'}", 
                      (x_offset, y_start + (line_height * 2)), d_color)

    # 4. FPS counter
    if fps > 0:
        draw_text_with_bg(overlay, f"FPS: {fps:.1f}", (x_offset, y_start + (line_height * 3)), (255, 255, 255))

    # 5. Smart Context & Sign Word
    ctx_mode = getattr(smart_result, "context_mode", "IDLE")
    ctx_color = (255, 255, 0) if ctx_mode == "CONVERSATION" else (0, 0, 255) if ctx_mode == "EMERGENCY" else (200, 200, 200)
    draw_text_with_bg(overlay, f"Mode: {ctx_mode}", (x_offset, y_start + (line_height * 4)), ctx_color)
    
    if sign_word and sign_word != "NONE":
        draw_text_with_bg(overlay, f"Sign: {sign_word}", (x_offset, y_start + (line_height * 5)), (0, 255, 255))

    # 6. System status (Bottom Left)
    draw_text_with_bg(overlay, f"VERS v{VERS_VERSION} | System Active", (x_offset, frame.shape[0] - 20), (220, 220, 220), alpha=0.4)

    if getattr(hand_results, "multi_hand_landmarks", None):
        h, w, _ = overlay.shape
        for hand_lms in hand_results.multi_hand_landmarks:
            # Draw standard skeleton connections
            mp_drawing.draw_landmarks(
                overlay,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            # Draw prominent fingertip halos
            for tip_id in [4, 8, 12, 16, 20]:
                lm = hand_lms.landmark[tip_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(overlay, (cx, cy), 7, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, (cx, cy), 9, (0, 165, 255), 2, cv2.LINE_AA)

            # Draw hand bounding box & floating gesture tag
            x_coords = [int(lm.x * w) for lm in hand_lms.landmark]
            y_coords = [int(lm.y * h) for lm in hand_lms.landmark]
            xmin, xmax = max(0, min(x_coords) - 15), min(w, max(x_coords) + 15)
            ymin, ymax = max(0, min(y_coords) - 15), min(h, max(y_coords) + 15)
            
            box_color = (0, 255, 0) if is_accepted else (0, 200, 255)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), box_color, 2, cv2.LINE_AA)
            
            if display_label != "NONE":
                badge_text = f" {display_label} ({display_confidence:.0%}) "
                (bw, bh), _ = cv2.getTextSize(badge_text, font, 0.6, 2)
                cv2.rectangle(overlay, (xmin, max(0, ymin - bh - 8)), (xmin + bw, ymin), box_color, -1)
                cv2.putText(overlay, badge_text, (xmin, max(bh + 2, ymin - 4)), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if face_lms is not None:
        mp_drawing.draw_landmarks(
            overlay,
            face_lms,
            mp_face.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
        )

    return overlay


def main() -> None:
    ensure_project_dirs()

    rprint("[bold blue]Starting VERS real-time demo[/bold blue]")
    model_bundle, labels = load_model()
    rprint(f"[cyan]Loaded classifier with gestures: {', '.join(labels)}[/cyan]")
    rprint("[cyan]Press 'q' to exit the OpenCV window.[/cyan]")

    cap, backend_info = open_camera_capture(max_index=4, warmup_reads=18)
    if cap is None:
        cap = cv2.VideoCapture(0)
        backend_info = "DEFAULT:0"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot access webcam. Check macOS permissions and close other camera apps."
        )

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize Smart Detector
    smart_detector = SmartDetector()
    
    # --- Sequence Sign Language Support ---
    sign_recognizer = SignLanguageRecognizer()
    sign_buffer = SequenceBuffer(window_size=30)
    intent_mapper = IntentMapper(memory_size=5)

    distress_history: deque[str] = deque(maxlen=DISTRESS_HISTORY_LIMIT)
    recent_preds: deque[tuple[str, float]] = deque(maxlen=SMOOTHING_WINDOW)
    last_alert_signature = ""
    last_alert_time = 0.0

    try:
        while True:
            try:
                frame_start_time = time.time()
                ok, frame = cap.read()
                if not ok:
                    log_error("Frame capture failed; skipping frame.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape

                hand_results = hands.process(rgb)
                face_results = face_mesh.process(rgb)
                
                # Extract vector for gesture tracker
                hand_vector = extract_hand_vector(hand_results)
                
                # Push raw vector to sign buffer (now works with DataFrame!)
                sign_buffer.push(hand_vector)

                # --- 1. Face Distress ---
                face_lms = (
                    face_results.multi_face_landmarks[0]
                    if getattr(face_results, "multi_face_landmarks", None)
                    else None
                )
                raw_distress = calc_distress(face_lms, width, height)

                # --- 2. Static Gesture Tracker ---
                gesture_label, gesture_confidence = "NONE", 0.0
                if hand_vector is not None:
                    gesture_label, gesture_confidence = predict_gesture(hand_vector)
                    
                # Smooth the static prediction
                recent_preds.append((gesture_label, gesture_confidence))
                smoothed_label, smoothed_confidence = smooth_prediction(recent_preds)

                # --- 3. Smart Detector ---
                # Pass flattened landmarks if available
                raw_lms = None
                if hand_results and getattr(hand_results, "multi_hand_landmarks", None):
                    if len(hand_results.multi_hand_landmarks) > 0:
                        raw_lms = np.array([[p.x, p.y, p.z] for p in hand_results.multi_hand_landmarks[0].landmark], dtype=np.float32).flatten()
                
                smart_result = smart_detector.update(
                    landmarks=raw_lms,
                    gesture_label=smoothed_label,
                    gesture_confidence=smoothed_confidence,
                    distress_score=raw_distress
                )
                
                distress_flag = smart_result.urgency_score > DISTRESS_THRESHOLD
                append_distress_history(distress_history, smart_result.urgency_score, distress_flag)

                # --- 4. LSTM Sign Language ---
                sign_word = "NONE"
                sign_conf = 0.0
                if sign_recognizer.available and sign_buffer.ready:
                    sign_word, sign_conf = sign_recognizer.predict(sign_buffer.get_tensor())
                    if sign_word != "NONE":
                        intent = intent_mapper.push_word(sign_word)
                        if intent:
                            payload = make_alert_payload(
                                intent.alert_type,
                                sign_conf,
                                raw_distress,
                                distress_flag,
                            )
                            payload["Message"] = intent.message
                            payload["Severity"] = intent.severity
                            
                            # Only alert for emergencies
                            if intent.is_emergency:
                                signature = f"SIGN:{intent.alert_type}:{intent.severity}"
                                now = time.time()
                                if signature != last_alert_signature or now - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                                    log_alert(payload)
                                    rprint(f"[bold red]SIGN ALERT:[/bold red] {intent.message}")
                                    last_alert_signature = signature
                                    last_alert_time = now
                                    if requests is not None:
                                        try:
                                            requests.post(ALERT_ENDPOINT, json=payload, timeout=0.6)
                                        except Exception:
                                            pass
                        # Clear buffer to prevent immediate repeated detections
                        sign_buffer.clear() 

                # --- 5. Regular Alerts ---
                now = time.time()
                if smart_result.gesture_label != "NONE" and smart_result.gesture_confidence >= HAND_CONF_THRESHOLD:
                    # Only alert if context is EMERGENCY or urgency is high
                    if smart_result.context_mode == "EMERGENCY" or smart_result.urgency_score > 0.4:
                        payload = make_alert_payload(
                            smart_result.gesture_label,
                            smart_result.gesture_confidence,
                            raw_distress,
                            distress_flag,
                        )
                        
                        signature = (
                            f"{payload['MainGesture']}:{payload['Severity']}:"
                            f"{payload['DistressFlag']}"
                        )
                        if signature != last_alert_signature or now - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                            log_alert(payload)
                            rprint(f"[bold red]ALERT:[/bold red] {json.dumps(payload, indent=2)}")
                            last_alert_signature = signature
                            last_alert_time = now
                            if requests is not None:
                                try:
                                    requests.post(ALERT_ENDPOINT, json=payload, timeout=0.6)
                                except Exception as exc:  # pragma: no cover - network/runtime path
                                    log_error(f"Alert POST failed: {exc}")

                fps = 1.0 / max(time.time() - frame_start_time, 1e-6)
                overlay = draw_overlay(
                    frame,
                    hand_results,
                    face_lms,
                    smart_result,
                    sign_word=sign_word,
                    fps=fps,
                )
                cv2.imshow("VERS Real-Time Demo (macOS)", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    rprint("[yellow]Exiting real-time loop.[/yellow]")
                    break

            except Exception as exc:
                log_error(f"Frame error [{backend_info}]: {exc}\n{traceback.format_exc()}")
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()

    rprint("[bold magenta]VERS demo finished.[/bold magenta]")


if __name__ == "__main__":
    main()
