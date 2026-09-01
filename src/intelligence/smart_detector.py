"""Smart AI Detection Engine for VERS v5.0.

Provides context-aware, multi-signal intelligence that goes beyond simple
gesture matching.  The SmartDetector fuses:

  1. **Motion Analysis** — hand velocity, acceleration, jitter (frantic = distress)
  2. **Gesture Repetition** — repeated same-sign = emphasis / urgency
  3. **Hold Duration** — how long a pose is held steady
  4. **Context Awareness** — automatic emergency vs. conversation mode switching
  5. **Adaptive Thresholds** — adjusts sensitivity based on recent activity

This module is intentionally stateful — it maintains a rolling window of
observations to build temporal context that single-frame classifiers miss.

Usage::

    detector = SmartDetector()
    # call every frame:
    result = detector.update(
        landmarks_63,       # raw hand landmarks
        gesture_label,      # from physics / LSTM classifier
        gesture_confidence, # 0.0 – 1.0
        distress_score,     # facial distress  0.0 – 1.0
    )
    print(result.context_mode)   # "EMERGENCY" or "CONVERSATION"
    print(result.urgency_score)  # 0.0 – 1.0  (motion-boosted)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("vers.intelligence.smart_detector")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SmartDetectionResult:
    """Immutable output of the smart detection engine."""

    # Core classification
    gesture_label: str = "NONE"
    gesture_confidence: float = 0.0

    # Motion features
    hand_velocity: float = 0.0       # pixels/sec (normalised)
    hand_acceleration: float = 0.0   # rate of velocity change
    hand_jitter: float = 0.0         # high-freq tremor (fear/stress indicator)
    hold_duration: float = 0.0       # seconds the current pose has been held

    # Smart analysis
    urgency_score: float = 0.0       # 0.0 – 1.0, combines motion + repetition + distress
    repetition_count: int = 0        # how many times current gesture repeated recently
    context_mode: str = "IDLE"       # EMERGENCY | CONVERSATION | IDLE
    is_repeated_gesture: bool = False
    is_frantic_motion: bool = False

    # Adaptive state
    sensitivity: float = 1.0         # current adaptive sensitivity multiplier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VELOCITY_WINDOW = 10          # frames for velocity calculation
REPETITION_WINDOW_SEC = 5.0   # seconds to count gesture repetitions
FRANTIC_VELOCITY_THRESHOLD = 0.15  # normalised velocity above this = frantic
HOLD_THRESHOLD = 0.02         # max motion to count as "holding" a pose
EMERGENCY_SIGNS = {
    "HELP", "MEDICAL", "FIRE", "POLICE", "AMBULANCE",
    "ACCIDENT", "DANGER", "EMERGENCY", "FALL", "STOP", "SOS",
}
CONVERSATION_SIGNS = {
    "HELLO", "THANK_YOU", "WATER", "FOOD", "PHONE",
    "FRIEND", "FAMILY", "GOOD", "BAD", "WANT", "SAFE",
}


# ---------------------------------------------------------------------------
# Smart Detector
# ---------------------------------------------------------------------------

class SmartDetector:
    """Context-aware multi-signal gesture intelligence engine.

    Call ``update()`` every frame with the latest hand landmarks and
    classifier output.  The detector builds temporal context and returns
    enriched detection results.

    Parameters
    ----------
    velocity_window : int
        Number of frames to average for velocity calculation.
    repetition_window : float
        Seconds within which repeated gestures are counted.
    """

    def __init__(
        self,
        velocity_window: int = VELOCITY_WINDOW,
        repetition_window: float = REPETITION_WINDOW_SEC,
    ) -> None:
        # Motion tracking
        self._landmark_history: deque[np.ndarray] = deque(maxlen=velocity_window)
        self._timestamp_history: deque[float] = deque(maxlen=velocity_window)
        self._velocity_history: deque[float] = deque(maxlen=velocity_window)

        # Gesture repetition tracking
        self._recent_gestures: deque[tuple[str, float]] = deque(maxlen=50)
        self._repetition_window = repetition_window

        # Hold detection
        self._hold_start: float = 0.0
        self._hold_label: str = "NONE"
        self._hold_active: bool = False

        # Context mode state machine
        self._context_mode: str = "IDLE"
        self._emergency_score_acc: float = 0.0
        self._conversation_score_acc: float = 0.0
        self._mode_decay: float = 0.95  # exponential decay per frame

        # Adaptive sensitivity
        self._activity_level: float = 0.0
        self._sensitivity: float = 1.0

        # Frame counter
        self._frame_count: int = 0

    def update(
        self,
        landmarks: Optional[np.ndarray],
        gesture_label: str,
        gesture_confidence: float,
        distress_score: float = 0.0,
    ) -> SmartDetectionResult:
        """Process one frame and return enriched detection result.

        Parameters
        ----------
        landmarks : ndarray or None
            Flattened 63-dim hand landmark vector, or None if no hand detected.
        gesture_label : str
            Output from the gesture classifier (e.g. "HELP", "HELLO", "NONE").
        gesture_confidence : float
            Classifier confidence (0.0 – 1.0).
        distress_score : float
            Facial distress score (0.0 – 1.0).

        Returns
        -------
        SmartDetectionResult
            Enriched detection with motion, context, and urgency analysis.
        """
        now = time.time()
        self._frame_count += 1

        # --- Motion Analysis ---
        velocity, acceleration, jitter = self._compute_motion(landmarks, now)

        # --- Hold Detection ---
        hold_duration = self._compute_hold(gesture_label, velocity, now)

        # --- Repetition Detection ---
        repetition_count, is_repeated = self._compute_repetition(gesture_label, now)

        # --- Frantic Motion Detection ---
        is_frantic = velocity > FRANTIC_VELOCITY_THRESHOLD * self._sensitivity

        # --- Context Mode (Emergency vs Conversation) ---
        context_mode = self._update_context_mode(
            gesture_label, gesture_confidence, is_frantic, distress_score, is_repeated
        )

        # --- Urgency Score ---
        urgency = self._compute_urgency(
            gesture_label, gesture_confidence, velocity, jitter,
            distress_score, repetition_count, is_frantic, hold_duration
        )

        # --- Adaptive Sensitivity ---
        self._update_sensitivity(velocity, gesture_label)

        return SmartDetectionResult(
            gesture_label=gesture_label,
            gesture_confidence=gesture_confidence,
            hand_velocity=velocity,
            hand_acceleration=acceleration,
            hand_jitter=jitter,
            hold_duration=hold_duration,
            urgency_score=urgency,
            repetition_count=repetition_count,
            context_mode=context_mode,
            is_repeated_gesture=is_repeated,
            is_frantic_motion=is_frantic,
            sensitivity=self._sensitivity,
        )

    # ------------------------------------------------------------------
    # Motion computation
    # ------------------------------------------------------------------

    def _compute_motion(
        self, landmarks: Optional[np.ndarray], now: float
    ) -> tuple[float, float, float]:
        """Compute hand velocity, acceleration, and jitter from landmark history."""
        if landmarks is None:
            return 0.0, 0.0, 0.0

        vec = np.asarray(landmarks, dtype=np.float32).flatten()
        if vec.shape[0] != 63:
            return 0.0, 0.0, 0.0

        self._landmark_history.append(vec)
        self._timestamp_history.append(now)

        if len(self._landmark_history) < 2:
            return 0.0, 0.0, 0.0

        # Velocity: Euclidean distance between consecutive frames / dt
        prev = self._landmark_history[-2]
        curr = self._landmark_history[-1]
        dt = max(now - self._timestamp_history[-2], 1e-6)
        displacement = float(np.linalg.norm(curr - prev))
        velocity = displacement / dt

        # Normalise velocity (typical webcam range)
        velocity = min(velocity / 10.0, 1.0)
        self._velocity_history.append(velocity)

        # Acceleration: rate of velocity change
        acceleration = 0.0
        if len(self._velocity_history) >= 2:
            acceleration = abs(self._velocity_history[-1] - self._velocity_history[-2]) / dt

        # Jitter: standard deviation of velocities in the window (tremor indicator)
        jitter = 0.0
        if len(self._velocity_history) >= 3:
            jitter = float(np.std(list(self._velocity_history)))

        return velocity, min(acceleration, 1.0), min(jitter, 1.0)

    # ------------------------------------------------------------------
    # Hold detection
    # ------------------------------------------------------------------

    def _compute_hold(self, label: str, velocity: float, now: float) -> float:
        """Detect how long a pose is being held steady."""
        if label == "NONE":
            self._hold_active = False
            self._hold_label = "NONE"
            return 0.0

        if velocity < HOLD_THRESHOLD:
            if not self._hold_active or self._hold_label != label:
                self._hold_start = now
                self._hold_label = label
                self._hold_active = True
            return now - self._hold_start
        else:
            self._hold_active = False
            return 0.0

    # ------------------------------------------------------------------
    # Repetition detection
    # ------------------------------------------------------------------

    def _compute_repetition(self, label: str, now: float) -> tuple[int, bool]:
        """Count how many times a gesture has been repeated in the window."""
        if label != "NONE":
            # Only count if it's a NEW occurrence (different from the last)
            if not self._recent_gestures or self._recent_gestures[-1][0] != label:
                self._recent_gestures.append((label, now))

        # Count occurrences of the current label within the window
        cutoff = now - self._repetition_window
        count = sum(
            1 for lbl, ts in self._recent_gestures
            if lbl == label and ts >= cutoff
        )

        is_repeated = count >= 3  # 3+ repetitions = emphasis
        return count, is_repeated

    # ------------------------------------------------------------------
    # Context mode (emergency vs conversation)
    # ------------------------------------------------------------------

    def _update_context_mode(
        self,
        label: str,
        confidence: float,
        is_frantic: bool,
        distress_score: float,
        is_repeated: bool,
    ) -> str:
        """State machine for automatic emergency/conversation switching."""
        # Decay previous scores
        self._emergency_score_acc *= self._mode_decay
        self._conversation_score_acc *= self._mode_decay

        # Accumulate based on current frame
        if label in EMERGENCY_SIGNS and confidence > 0.3:
            boost = confidence * 0.3
            if is_frantic:
                boost *= 1.5
            if distress_score > 0.05:
                boost *= 1.3
            if is_repeated:
                boost *= 1.2
            self._emergency_score_acc += boost

        elif label in CONVERSATION_SIGNS and confidence > 0.3:
            self._conversation_score_acc += confidence * 0.25

        # Determine mode
        if self._emergency_score_acc > 0.4:
            self._context_mode = "EMERGENCY"
        elif self._conversation_score_acc > 0.3:
            self._context_mode = "CONVERSATION"
        elif max(self._emergency_score_acc, self._conversation_score_acc) < 0.05:
            self._context_mode = "IDLE"

        return self._context_mode

    # ------------------------------------------------------------------
    # Urgency score
    # ------------------------------------------------------------------

    def _compute_urgency(
        self,
        label: str,
        confidence: float,
        velocity: float,
        jitter: float,
        distress_score: float,
        repetition_count: int,
        is_frantic: bool,
        hold_duration: float,
    ) -> float:
        """Compute a unified urgency score from all signals.

        Weights:
          - Gesture confidence:  30%
          - Motion urgency:      20%  (velocity + jitter)
          - Distress (face):     20%
          - Repetition boost:    15%
          - Hold duration:       15%  (longer hold = more deliberate)
        """
        if label == "NONE":
            return 0.0

        # Base from gesture confidence
        w_conf = confidence * 0.30

        # Motion urgency (fast + jittery = panicked)
        motion_signal = min((velocity * 0.6 + jitter * 0.4), 1.0)
        w_motion = motion_signal * 0.20

        # Facial distress
        w_distress = min(distress_score / 0.12, 1.0) * 0.20

        # Repetition (capped at 5x)
        rep_factor = min(repetition_count / 5.0, 1.0)
        w_rep = rep_factor * 0.15

        # Hold duration (longer = more deliberate, cap at 3s)
        hold_factor = min(hold_duration / 3.0, 1.0)
        w_hold = hold_factor * 0.15

        urgency = w_conf + w_motion + w_distress + w_rep + w_hold

        # Emergency context boost
        if label in EMERGENCY_SIGNS:
            urgency *= 1.15

        # Frantic motion critical boost
        if is_frantic and label in EMERGENCY_SIGNS:
            urgency *= 1.25

        return min(urgency, 1.0)

    # ------------------------------------------------------------------
    # Adaptive sensitivity
    # ------------------------------------------------------------------

    def _update_sensitivity(self, velocity: float, label: str) -> None:
        """Adjust detection sensitivity based on activity level.

        High activity (lots of movement) → slightly raise thresholds to
        reduce false positives.  Low activity → lower thresholds for
        quicker detection.
        """
        # Exponential moving average of activity
        self._activity_level = 0.95 * self._activity_level + 0.05 * velocity

        # Map activity to sensitivity (inverse relationship)
        if self._activity_level > 0.3:
            self._sensitivity = 1.2  # raise thresholds (noisy environment)
        elif self._activity_level < 0.05:
            self._sensitivity = 0.8  # lower thresholds (quiet, be responsive)
        else:
            self._sensitivity = 1.0

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state."""
        self._landmark_history.clear()
        self._timestamp_history.clear()
        self._velocity_history.clear()
        self._recent_gestures.clear()
        self._hold_active = False
        self._hold_label = "NONE"
        self._context_mode = "IDLE"
        self._emergency_score_acc = 0.0
        self._conversation_score_acc = 0.0
        self._activity_level = 0.0
        self._sensitivity = 1.0
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count
