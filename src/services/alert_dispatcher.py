"""Alert Dispatcher service for VERS v3.0.

Responsible for:
  1. Packaging raw fusion results into structured JSON alert payloads
  2. Simulating GPS location data for dashboard map rendering
  3. Managing alert cooldown logic to prevent duplicate spam
  4. Forwarding payloads to the webhook endpoint and TTS engine
  5. [v3.0] Multi-channel notification: TTS, mock SMS, mock Email
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import sys

# Ensure src can be imported when running standalone
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.services import voice_tts

logger = logging.getLogger("vers.services.alert_dispatcher")

# ---------------------------------------------------------------------------
# Notification channel configuration (via VERS_NOTIFY_CHANNELS env var)
# ---------------------------------------------------------------------------
_ACTIVE_CHANNELS: set[str] = set(
    os.environ.get("VERS_NOTIFY_CHANNELS", "tts").lower().split(",")
)

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
EMAIL_OUTBOX = LOG_DIR / "email_outbox"

# ---------------------------------------------------------------------------
# Simulated location data
# ---------------------------------------------------------------------------
_SIMULATED_LOCATIONS: list[dict[str, Any]] = [
    {"lat": 28.6139, "lon": 77.2090, "label": "New Delhi HQ"},
    {"lat": 28.5355, "lon": 77.3910, "label": "Noida Sector 62"},
    {"lat": 28.4595, "lon": 77.0266, "label": "Gurgaon Campus"},
    {"lat": 28.7041, "lon": 77.1025, "label": "Delhi University"},
    {"lat": 28.6304, "lon": 77.2177, "label": "India Gate Area"},
]


def _simulated_location() -> dict[str, Any]:
    """Pick a random simulated GPS coordinate."""
    loc = random.choice(_SIMULATED_LOCATIONS)
    return {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "label": loc["label"],
        "source": "simulated",
    }


# ---------------------------------------------------------------------------
# Cooldown state
# ---------------------------------------------------------------------------
_last_signature: str = ""
_last_alert_time: float = 0.0
ALERT_COOLDOWN_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Notification channel implementations
# ---------------------------------------------------------------------------

def _notify_sms(payload: dict[str, Any]) -> None:
    """Mock SMS dispatch — logs to logs/sms_dispatch.log."""
    recipient = os.environ.get("VERS_SMS_RECIPIENT", "+91-9999999999")
    sms_body = (
        f"[VERS ALERT] {payload['ThreatLevel']} — "
        f"{payload['MainGesture']} detected. "
        f"Severity: {payload['SeverityScore']:.2f}. "
        f"Location: {payload.get('Location', {}).get('label', 'Unknown')}. "
        f"Time: {payload['Timestamp'][:19]}"
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sms_log = LOG_DIR / "sms_dispatch.log"
    with sms_log.open("a", encoding="utf-8") as f:
        f.write(f"TO: {recipient} | {sms_body}\n")

    logger.info("SMS dispatched to %s: %s", recipient, payload["MainGesture"])


def _notify_email(payload: dict[str, Any]) -> None:
    """Mock email dispatch — generates .eml files in logs/email_outbox/."""
    recipient = os.environ.get("VERS_EMAIL_RECIPIENT", "admin@emergency.local")
    EMAIL_OUTBOX.mkdir(parents=True, exist_ok=True)

    timestamp_safe = payload["Timestamp"].replace(":", "-")[:19]
    eml_path = EMAIL_OUTBOX / f"alert_{timestamp_safe}.eml"

    eml_content = (
        f"From: vers-alerts@system.local\n"
        f"To: {recipient}\n"
        f"Subject: [VERS {payload['ThreatLevel']}] {payload['MainGesture']} Alert\n"
        f"Date: {payload['Timestamp']}\n"
        f"Content-Type: text/plain; charset=utf-8\n"
        f"\n"
        f"VERS Emergency Response System — Alert Notification\n"
        f"{'=' * 50}\n"
        f"\n"
        f"Threat Level: {payload['ThreatLevel']}\n"
        f"Gesture:      {payload['MainGesture']}\n"
        f"Emotion:      {payload.get('DominantEmotion', 'N/A')}\n"
        f"Severity:     {payload['SeverityScore']:.4f}\n"
        f"Distress:     {'YES' if payload.get('DistressFlag') else 'NO'}\n"
        f"Location:     {payload.get('Location', {}).get('label', 'Unknown')}\n"
        f"\n"
        f"Full Payload:\n"
        f"{json.dumps(payload, indent=2)}\n"
    )

    eml_path.write_text(eml_content, encoding="utf-8")
    logger.info("Email generated: %s", eml_path.name)


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_alert_payload(
    gesture_label: str,
    gesture_confidence: float,
    dominant_emotion: str,
    emotion_distress: float,
    severity_score: float,
    threat_level: str,
    distress_flag: bool,
) -> dict[str, Any]:
    """Construct a fully structured JSON alert payload."""
    location = _simulated_location()

    severity_map = {
        "CRITICAL": "Critical",
        "HIGH": "High",
        "MEDIUM": "Medium",
        "LOW": "Low",
        "NONE": "None",
    }

    message_map = {
        "SOS": "SOS distress signal detected — immediate assistance required.",
        "EMERGENCY": "Emergency gesture recognised — dispatching alert.",
        "ACCIDENT": "Accident indicator detected — medical team notified.",
        "MEDICAL": "Medical assistance gesture detected — standby.",
        "SAFE": "Safe gesture acknowledged — situation under control.",
        "FIRE": "Fire emergency reported — fire services alerted.",
        "POLICE": "Police assistance requested.",
        "AMBULANCE": "Ambulance requested — critical medical emergency.",
        "DANGER": "Danger signal detected.",
        "HELP": "Help requested via sign language.",
        "PAIN": "Pain indicated via sign language.",
        "FALL": "Fall incident reported.",
        "STOP": "Stop signal detected.",
        "YES": "Affirmative response.",
        "NO": "Negative response.",
        "PLEASE": "Polite request modifier.",
        # Conversational
        "HELLO": "Hello — greeting detected.",
        "THANK_YOU": "Thank you gesture detected.",
        "WATER": "Water request detected.",
        "FOOD": "Food request detected.",
        "FAMILY": "Family reference detected.",
        "FRIEND": "Friend reference detected.",
        "WHERE": "Location inquiry detected.",
        "WANT": "Want/need expressed.",
        "UNDERSTAND": "Understanding confirmed.",
        "NAME": "Name inquiry detected.",
        "SORRY": "Apology detected.",
        "GOOD": "Positive status indicated.",
        "BAD": "Negative status indicated.",
        "MORE": "More requested.",
        "FINISHED": "Finished/done indicated.",
        "PHONE": "Requesting a phone call / Phone mentioned.",
    }

    return {
        "Timestamp": datetime.now().isoformat() + "Z",
        "SystemVersion": "VERS-5.0-Production",
        "MainGesture": gesture_label,
        "GestureConfidence": round(gesture_confidence, 4),
        "DominantEmotion": dominant_emotion,
        "EmotionDistress": round(emotion_distress, 4),
        "SeverityScore": round(severity_score, 4),
        "ThreatLevel": threat_level,
        "Severity": severity_map.get(threat_level, "Low"),
        "DistressFlag": distress_flag,
        "Message": message_map.get(gesture_label, "Unknown gesture detected."),
        "Location": location,
    }


# ---------------------------------------------------------------------------
# Primary dispatch function
# ---------------------------------------------------------------------------

_last_alert_by_gesture: dict[str, float] = {}
_global_last_alert_time: float = 0.0
GLOBAL_ALERT_COOLDOWN_SECONDS: float = 3.0
SAME_GESTURE_COOLDOWN_SECONDS: float = 6.0


def dispatch(
    gesture_label: str,
    gesture_confidence: float,
    dominant_emotion: str,
    emotion_distress: float,
    severity_score: float,
    threat_level: str,
    distress_flag: bool,
    *,
    enable_tts: bool = True,
    force: bool = False,
) -> Optional[dict[str, Any]]:
    """Build and dispatch an alert if cooldown allows; returns payload or None."""
    global _global_last_alert_time, _last_alert_by_gesture

    if gesture_label == "NONE" or not gesture_label:
        return None

    now = time.time()

    if not force:
        last_same = _last_alert_by_gesture.get(gesture_label, 0.0)
        if (now - last_same) < SAME_GESTURE_COOLDOWN_SECONDS:
            return None  # Suppressed by same-gesture cooldown
        if (now - _global_last_alert_time) < GLOBAL_ALERT_COOLDOWN_SECONDS:
            return None  # Suppressed by global cooldown

    payload = build_alert_payload(
        gesture_label=gesture_label,
        gesture_confidence=gesture_confidence,
        dominant_emotion=dominant_emotion,
        emotion_distress=emotion_distress,
        severity_score=severity_score,
        threat_level=threat_level,
        distress_flag=distress_flag,
    )

    _last_alert_by_gesture[gesture_label] = now
    _global_last_alert_time = now

    logger.info("ALERT dispatched: %s [%s]", gesture_label, threat_level)

    # --- Multi-channel notification (v3.0) ---
    # Skip emergency notifications for conversational (Info) gestures
    _INFO_GESTURES = {
        "HELLO", "THANK_YOU", "PLEASE", "YES", "NO",
        "WATER", "FOOD", "FAMILY", "FRIEND",
        "WHERE", "WANT", "UNDERSTAND", "NAME", "SORRY", "GOOD",
        "BAD", "MORE", "FINISHED", "PHONE", "SAFE",
    }
    is_conversational = gesture_label in _INFO_GESTURES

    if enable_tts and "tts" in _ACTIVE_CHANNELS and not is_conversational:
        voice_tts.speak(
            f"{gesture_label} detected. Threat level {threat_level}."
        )

    if "sms" in _ACTIVE_CHANNELS and threat_level in ("CRITICAL", "HIGH") and not is_conversational:
        try:
            _notify_sms(payload)
        except Exception as exc:
            logger.debug("SMS notification failed: %s", exc)

    if "email" in _ACTIVE_CHANNELS:
        try:
            _notify_email(payload)
        except Exception as exc:
            logger.debug("Email notification failed: %s", exc)

    return payload

if __name__ == "__main__":
    # Configure logging for standalone test
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Alert Dispatcher")
    
    # Test an emergency gesture
    print("Dispatching SOS Alert...")
    dispatch(
        gesture_label="SOS",
        gesture_confidence=0.95,
        dominant_emotion="FEAR",
        emotion_distress=0.88,
        severity_score=0.92,
        threat_level="CRITICAL",
        distress_flag=True,
        enable_tts=True
    )
    
    print("\nDispatching conversational gesture (should skip TTS)...")
    dispatch(
        gesture_label="HELLO",
        gesture_confidence=0.90,
        dominant_emotion="HAPPY",
        emotion_distress=0.10,
        severity_score=0.20,
        threat_level="LOW",
        distress_flag=False,
        enable_tts=True
    )
