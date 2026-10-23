"""Offline, bounded admission of speech results to a future intent classifier."""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from zentomic.gather import GatherDecision, resolve_gather


MAX_TRANSCRIPT_CHARS = 2000
MAX_TRANSCRIPT_UTF8_BYTES = MAX_TRANSCRIPT_CHARS * 4
MAX_CONFIDENCE_CHARACTERS = 32
_CONFIDENCE = re.compile(r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)", re.ASCII)


@dataclass(frozen=True)
class SpeechDecision:
    """Next action, optional transcript/target, and completed collection count.

    Omit caller text from repr/str to reduce accidental diagnostic disclosure.
    Explicit transcript access and serialization still require privacy controls.
    """

    action: Literal["classify", "retry", "fallback"]
    transcript: str | None = field(repr=False)
    target: str | None
    attempts: int


def _validate_speech_configuration(
    *, fallback_target: str, attempts: int, max_attempts: int,
    minimum_confidence_percent: int | None = None,
    require_confidence: bool = False,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
) -> GatherDecision:
    """Return the collection budget after validating trusted speech policy."""
    if (minimum_confidence_percent is not None
            and (type(minimum_confidence_percent) is not int
                 or not 0 <= minimum_confidence_percent <= 100)):
        raise ValueError("minimum_confidence_percent must be an integer from 0 to 100 or None")
    if type(require_confidence) is not bool:
        raise ValueError("require_confidence must be a boolean")
    if (type(max_transcript_characters) is not int
            or not 1 <= max_transcript_characters <= MAX_TRANSCRIPT_CHARS):
        raise ValueError(
            "max_transcript_characters must be an integer from 1 to 2000"
        )
    if (type(max_transcript_utf8_bytes) is not int
            or not 1 <= max_transcript_utf8_bytes <= MAX_TRANSCRIPT_UTF8_BYTES):
        raise ValueError(
            "max_transcript_utf8_bytes must be an integer from 1 to 8000"
        )
    return resolve_gather(
        None, {}, fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts,
    )


def _confidence_meets_minimum(
    confidence: str | None, minimum_confidence_percent: int | None,
    *, require_confidence: bool = False,
) -> bool:
    """Apply optional trusted presence and threshold policy to a provider score."""
    # Twilio does not guarantee Confidence is present. Preserve admission when
    # it is absent unless trusted configuration explicitly requires a score.
    if minimum_confidence_percent is None and not require_confidence:
        return True
    if confidence is None:
        return not require_confidence
    if (not isinstance(confidence, str)
            or len(confidence) > MAX_CONFIDENCE_CHARACTERS
            or _CONFIDENCE.fullmatch(confidence) is None):
        return False
    if minimum_confidence_percent is None:
        return True
    return Decimal(confidence) * 100 >= minimum_confidence_percent


def _resolve_validated_speech(
    speech: str | None, *, budget: GatherDecision,
    attempts: int, max_attempts: int, confidence: str | None = None,
    minimum_confidence_percent: int | None = None,
    require_confidence: bool = False,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
) -> SpeechDecision:
    """Admit speech after the caller validated and snapshotted its budget."""
    if (attempts < max_attempts and isinstance(speech, str)
            and len(speech) <= max_transcript_characters and speech.strip()
            and _confidence_meets_minimum(
                confidence, minimum_confidence_percent,
                require_confidence=require_confidence,
            )):
        try:
            encoded_speech = speech.encode("utf-8")
        except UnicodeEncodeError:
            pass  # Malformed text follows the same finite retry budget.
        else:
            if len(encoded_speech) <= max_transcript_utf8_bytes:
                return SpeechDecision("classify", speech, None, budget.attempts)
    action = "retry" if budget.action == "retry" else "fallback"
    return SpeechDecision(action, None, budget.target, budget.attempts)


def resolve_speech_gather(
    speech: str | None, *, fallback_target: str, attempts: int, max_attempts: int = 3,
    confidence: str | None = None, minimum_confidence_percent: int | None = None,
    require_confidence: bool = False,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
) -> SpeechDecision:
    """Admit bounded nonblank text, or retry silence within a trusted budget.

    Reuse the keypad collection budget and configuration validation. Missing,
    malformed, blank, oversized, or explicitly low-confidence results consume
    one attempt but never reach a classifier. A configured confidence threshold
    is an integer percentage from 0 through 100. Missing provider confidence
    remains admissible because the field is not guaranteed, unless trusted
    configuration explicitly requires it. Under either confidence policy, a
    present score must be a bounded decimal from 0 through 1. The trusted
    transcript limits are integers from 1 through the fixed 2000-character and
    8000-byte ceilings and apply before whitespace trimming. Byte length is
    measured from strict UTF-8 encoding; accepted text is returned unchanged.
    Surrogate code points are rejected, not silently replaced or repaired.
    Valid text on the last attempt may be classified, but an already exhausted
    budget always falls back.

    Input must come from an authenticated callback bound to this speech step.
    Persist/deduplicate the decision before classification or reprompting.
    Text is still untrusted: this is not prompt-injection protection, intent
    classification, confirmation, authorization, or a model token/cost limit.
    """
    budget = _validate_speech_configuration(
        fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts,
        minimum_confidence_percent=minimum_confidence_percent,
        require_confidence=require_confidence,
        max_transcript_characters=max_transcript_characters,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
    )
    return _resolve_validated_speech(
        speech, budget=budget, attempts=attempts, max_attempts=max_attempts,
        confidence=confidence,
        minimum_confidence_percent=minimum_confidence_percent,
        require_confidence=require_confidence,
        max_transcript_characters=max_transcript_characters,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
    )
