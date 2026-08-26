"""LSTM-based sign language classifier for VERS v5.0.

Recognises emergency AND conversational sign language words from sequences
of MediaPipe hand landmarks.  The model is a lightweight 2-layer LSTM that
processes 30-frame sequences (≈1 second) and outputs per-word probabilities.

PyTorch is an *optional* dependency.  If unavailable the module exposes a
graceful fallback that always returns ``("NONE", 0.0)``.

Architecture::

    Input (batch, 30, 63) → LSTM(256, 2 layers) → Dense(128) → ReLU → Dense(num_classes) → Softmax

Trained weights are loaded from ``models/sign_language_lstm.pth``.

v5.0 Changes:
  - Expanded vocabulary from 16 to 31 words (15 emergency + 15 conversational + NONE)
  - Wider hidden layer (256) for richer representation
  - Added attention mechanism for better temporal focus
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("vers.vision.sign_language")

# ---------------------------------------------------------------------------
# Emergency vocabulary — Phase 1 (15 words)
# ---------------------------------------------------------------------------
EMERGENCY_SIGNS: list[str] = [
    "HELP",
    "STOP",
    "ACCIDENT",
    "MEDICAL",
    "FIRE",
    "POLICE",
    "AMBULANCE",
    "DANGER",
    "PAIN",
    "FALL",
    "SAFE",
    "YES",
    "NO",
    "PLEASE",
    "EMERGENCY",
]

# ---------------------------------------------------------------------------
# Conversational vocabulary — Phase 2 (15 words)
# ---------------------------------------------------------------------------
CONVERSATIONAL_SIGNS: list[str] = [
    "HELLO",
    "THANK_YOU",
    "WATER",
    "FOOD",
    "FAMILY",
    "FRIEND",
    "WHERE",
    "WANT",
    "UNDERSTAND",
    "NAME",
    "SORRY",
    "GOOD",
    "BAD",
    "MORE",
    "FINISHED",
]

# Full vocabulary = NONE + emergency + conversational
SIGN_VOCABULARY: list[str] = ["NONE"] + EMERGENCY_SIGNS + CONVERSATIONAL_SIGNS

NUM_CLASSES = len(SIGN_VOCABULARY)
SEQUENCE_LENGTH = 30
NUM_FEATURES = 63
HIDDEN_SIZE = 256
NUM_LAYERS = 2

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "sign_language_lstm.pth"

# ---------------------------------------------------------------------------
# Try to import PyTorch — graceful fallback if unavailable
# ---------------------------------------------------------------------------
_torch = None
_nn = None

try:
    import torch
    import torch.nn as nn
    _torch = torch
    _nn = nn
except ImportError:
    logger.info("PyTorch not installed — sign language recognition disabled.")


# ---------------------------------------------------------------------------
# Model definition with temporal attention
# ---------------------------------------------------------------------------

if _nn is not None:
    class SignLanguageLSTM(_nn.Module):
        """LSTM with temporal attention for sign language word classification.

        The attention mechanism learns which frames in the sequence are most
        informative, allowing the model to focus on the 'peak' of a gesture
        rather than treating all 30 frames equally.
        """

        def __init__(
            self,
            num_features: int = NUM_FEATURES,
            hidden_size: int = HIDDEN_SIZE,
            num_layers: int = NUM_LAYERS,
            num_classes: int = NUM_CLASSES,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.lstm = _nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # Temporal attention: learn which frames matter most
            self.attention = _nn.Sequential(
                _nn.Linear(hidden_size, 64),
                _nn.Tanh(),
                _nn.Linear(64, 1),
            )
            self.classifier = _nn.Sequential(
                _nn.Linear(hidden_size, 128),
                _nn.ReLU(),
                _nn.Dropout(dropout),
                _nn.Linear(128, 64),
                _nn.ReLU(),
                _nn.Dropout(dropout * 0.5),
                _nn.Linear(64, num_classes),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

            # Attention weights over time steps
            attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = _torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

            # Weighted sum of LSTM outputs
            context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_size)

            logits = self.classifier(context)
            return logits
else:
    SignLanguageLSTM = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class SignLanguageRecognizer:
    """High-level interface for sign language word prediction.

    Handles model loading, inference, and confidence thresholding.
    """

    def __init__(self, confidence_threshold: float = 0.55) -> None:
        self._model: Optional[object] = None
        self._available = False
        self._confidence_threshold = confidence_threshold
        self._load_model()

    def _load_model(self) -> None:
        if _torch is None or SignLanguageLSTM is None:
            logger.info("Sign language model unavailable (PyTorch not installed).")
            return

        self._model = SignLanguageLSTM()

        if MODEL_PATH.exists():
            try:
                state_dict = _torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
                self._model.load_state_dict(state_dict)
                self._model.eval()
                self._available = True
                logger.info("Sign language LSTM loaded from %s (%d classes)", MODEL_PATH, NUM_CLASSES)
            except Exception as exc:
                logger.warning("Failed to load sign language model: %s", exc)
                self._available = False
        else:
            # Model defined but no trained weights — keep the architecture
            # available for training but mark inference as unavailable
            self._model.eval()
            self._available = False
            logger.info(
                "Sign language LSTM architecture ready, but no trained weights "
                "found at %s. Run training first.",
                MODEL_PATH,
            )

    @property
    def available(self) -> bool:
        """Whether a trained model is loaded and ready for inference."""
        return self._available

    @property
    def vocabulary(self) -> list[str]:
        """Return the full vocabulary list."""
        return SIGN_VOCABULARY

    def predict(self, sequence: np.ndarray) -> tuple[str, float]:
        """Predict a sign language word from a (1, 30, 63) sequence array.

        Returns ``(word, confidence)`` or ``("NONE", 0.0)`` if unavailable.
        """
        if not self._available or _torch is None:
            return "NONE", 0.0

        try:
            tensor = _torch.tensor(sequence, dtype=_torch.float32)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)  # Add batch dimension

            with _torch.no_grad():
                logits = self._model(tensor)
                probs = _torch.softmax(logits, dim=-1)
                confidence, idx = probs.max(dim=-1)
                conf_val = float(confidence.item())
                class_idx = int(idx.item())

            if conf_val < self._confidence_threshold:
                return "NONE", conf_val

            word = SIGN_VOCABULARY[class_idx] if class_idx < len(SIGN_VOCABULARY) else "NONE"
            return word, conf_val

        except Exception as exc:
            logger.debug("Sign language prediction failed: %s", exc)
            return "NONE", 0.0

    def predict_top_k(self, sequence: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k predictions with confidence scores.

        Useful for the smart detector to consider multiple interpretations.
        """
        if not self._available or _torch is None:
            return [("NONE", 0.0)]

        try:
            tensor = _torch.tensor(sequence, dtype=_torch.float32)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)

            with _torch.no_grad():
                logits = self._model(tensor)
                probs = _torch.softmax(logits, dim=-1)
                topk_probs, topk_indices = probs.topk(min(k, NUM_CLASSES), dim=-1)

            results = []
            for i in range(topk_probs.shape[-1]):
                conf = float(topk_probs[0, i].item())
                idx = int(topk_indices[0, i].item())
                word = SIGN_VOCABULARY[idx] if idx < len(SIGN_VOCABULARY) else "NONE"
                results.append((word, conf))

            return results

        except Exception as exc:
            logger.debug("Top-k prediction failed: %s", exc)
            return [("NONE", 0.0)]

    @staticmethod
    def is_emergency_sign(word: str) -> bool:
        """Check if a word belongs to the emergency vocabulary."""
        return word in EMERGENCY_SIGNS

    @staticmethod
    def is_conversational_sign(word: str) -> bool:
        """Check if a word belongs to the conversational vocabulary."""
        return word in CONVERSATIONAL_SIGNS
