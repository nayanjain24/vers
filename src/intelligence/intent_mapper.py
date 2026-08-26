"""NLP-lite intent interpreter for sign language word sequences.

Receives a rolling buffer of predicted sign language words and maps
recognised patterns to structured emergency intents.  This is a rule-based
system (no external NLP model required) designed for the 30-word vocabulary
(15 emergency + 15 conversational).

v5.0 — Added conversational intent rules alongside emergency patterns.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("vers.intelligence.intent_mapper")


@dataclass(frozen=True)
class EmergencyIntent:
    """Structured output of the intent interpretation engine."""

    alert_type: str       # ACCIDENT, MEDICAL, FIRE, GREETING, REQUEST, etc.
    severity: str         # CRITICAL, HIGH, MEDIUM, LOW, INFO, NONE
    message: str          # Human-readable description
    source_words: list[str]  # The words that triggered this intent
    is_emergency: bool = True  # False for conversational intents


# ---------------------------------------------------------------------------
# Pattern rules — ordered by priority (first match wins)
# ---------------------------------------------------------------------------

_INTENT_RULES: list[tuple[set[str], str, str, str, bool]] = [
    # === Critical emergencies ===
    ({"FIRE"},                "FIRE",      "CRITICAL", "Fire emergency reported via sign language.", True),
    ({"ACCIDENT", "HELP"},    "ACCIDENT",  "CRITICAL", "Accident with help request — immediate response needed.", True),
    ({"EMERGENCY", "HELP"},   "EMERGENCY", "CRITICAL", "Emergency with help request detected.", True),
    ({"AMBULANCE"},           "MEDICAL",   "CRITICAL", "Ambulance requested via sign language.", True),
    # === High severity ===
    ({"ACCIDENT"},            "ACCIDENT",  "HIGH",     "Accident reported via sign language.", True),
    ({"EMERGENCY"},           "EMERGENCY", "HIGH",     "Emergency situation indicated.", True),
    ({"MEDICAL", "PAIN"},     "MEDICAL",   "HIGH",     "Medical assistance needed — pain indicated.", True),
    ({"MEDICAL"},             "MEDICAL",   "HIGH",     "Medical assistance requested.", True),
    ({"POLICE"},              "POLICE",    "HIGH",     "Police assistance requested via sign language.", True),
    ({"DANGER"},              "EMERGENCY", "HIGH",     "Danger indicated via sign language.", True),
    ({"HELP"},                "SOS",       "HIGH",     "Help requested via sign language.", True),
    ({"FALL", "PAIN"},        "MEDICAL",   "HIGH",     "Fall with pain reported.", True),
    ({"HELP", "FAMILY"},      "EMERGENCY", "HIGH",     "Family emergency — help needed.", True),
    # === Medium severity ===
    ({"FALL"},                "ACCIDENT",  "MEDIUM",   "Fall incident reported.", True),
    ({"STOP"},                "EMERGENCY", "MEDIUM",   "Stop signal detected.", True),
    ({"PAIN"},                "MEDICAL",   "MEDIUM",   "Pain indicated via sign language.", True),
    # === Low severity ===
    ({"SAFE"},                "SAFE",      "LOW",      "Safe status confirmed via sign language.", True),
    ({"YES"},                 "CONFIRM",   "LOW",      "Affirmative response received.", True),
    ({"NO"},                  "DENY",      "LOW",      "Negative response received.", True),
    # === Conversational (INFO level — no emergency) ===
    ({"WATER", "PLEASE"},     "WATER",     "INFO",     "Water requested.", False),
    ({"FOOD", "PLEASE"},      "FOOD",      "INFO",     "Food requested.", False),
    ({"WATER", "WANT"},       "WATER",     "INFO",     "Water requested.", False),
    ({"FOOD", "WANT"},        "FOOD",      "INFO",     "Food requested.", False),
    ({"HELLO"},               "HELLO",     "INFO",     "Hello — greeting received.", False),
    ({"THANK_YOU"},           "THANK_YOU", "INFO",     "Thank you received.", False),
    ({"SORRY"},               "SORRY",     "INFO",     "Apology received.", False),
    ({"GOOD"},                "GOOD",      "INFO",     "Positive status indicated.", False),
    ({"BAD"},                 "BAD",       "INFO",     "Negative status indicated.", False),
    ({"UNDERSTAND"},          "UNDERSTAND","INFO",     "Understanding confirmed.", False),
    ({"NAME"},                "NAME",      "INFO",     "Name inquiry detected.", False),
    ({"WHERE"},               "WHERE",     "INFO",     "Location inquiry detected.", False),
    ({"WANT"},                "WANT",      "INFO",     "Request or want indicated.", False),
    ({"MORE"},                "MORE",      "INFO",     "More requested.", False),
    ({"FINISHED"},            "FINISHED",  "INFO",     "Finished / done indicated.", False),
    ({"WATER"},               "WATER",     "INFO",     "Water indicated.", False),
    ({"FOOD"},                "FOOD",      "INFO",     "Food indicated.", False),
    ({"FAMILY"},              "FAMILY",    "INFO",     "Family mentioned.", False),
    ({"FRIEND"},              "FRIEND",    "INFO",     "Friend mentioned.", False),
]


class IntentMapper:
    """Maps rolling sign language word predictions to emergency intents.

    Maintains a short-term word memory (default 5 words) and checks against
    the pattern rules on each new word.
    """

    def __init__(self, memory_size: int = 5) -> None:
        self._memory: deque[str] = deque(maxlen=memory_size)

    def push_word(self, word: str) -> Optional[EmergencyIntent]:
        """Add a recognised word and attempt to match an intent pattern.

        Returns an ``EmergencyIntent`` if a pattern matches, otherwise ``None``.
        Words equal to ``"NONE"`` are ignored.
        """
        if word == "NONE" or not word:
            return None

        self._memory.append(word)
        current_words = set(self._memory)

        for required, alert_type, severity, message, is_emergency in _INTENT_RULES:
            if required.issubset(current_words):
                intent = EmergencyIntent(
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                    source_words=list(self._memory),
                    is_emergency=is_emergency,
                )
                logger.info(
                    "Intent matched: %s [%s] from words %s (emergency=%s)",
                    alert_type, severity, list(self._memory), is_emergency,
                )
                # Clear memory after a match to avoid re-triggering
                self._memory.clear()
                return intent

        return None

    def reset(self) -> None:
        """Clear the word memory."""
        self._memory.clear()

    @property
    def current_words(self) -> list[str]:
        """The current words in memory (for display)."""
        return list(self._memory)
