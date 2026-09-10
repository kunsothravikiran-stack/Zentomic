"""Offline, bounded admission of speech results to a future intent classifier."""

from dataclasses import dataclass, field
from typing import Literal

from zentomic.gather import resolve_gather


MAX_TRANSCRIPT_CHARS = 2000


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


def resolve_speech_gather(
    speech: str | None, *, fallback_target: str, attempts: int, max_attempts: int = 3,
) -> SpeechDecision:
    """Admit bounded nonblank text, or retry silence within a trusted budget.

    Reuse the keypad collection budget and configuration validation. Missing,
    malformed, blank, or oversized results consume one attempt but never reach
    a classifier. The 2000-character limit applies before whitespace trimming;
    accepted text must be UTF-8 encodable and is returned unchanged. Surrogate
    code points are rejected, not silently replaced or repaired.
    Valid text on the last attempt may be classified, but an already exhausted
    budget always falls back.

    Input must come from an authenticated callback bound to this speech step.
    Persist/deduplicate the decision before classification or reprompting.
    Text is still untrusted: this is not prompt-injection protection, intent
    classification, confirmation, authorization, or a model token/cost limit.
    """
    budget = resolve_gather(
        None, {}, fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts,
    )
    if (attempts < max_attempts and isinstance(speech, str)
            and len(speech) <= MAX_TRANSCRIPT_CHARS and speech.strip()):
        try:
            speech.encode("utf-8")
        except UnicodeEncodeError:
            pass  # Malformed text follows the same finite retry budget.
        else:
            return SpeechDecision("classify", speech, None, budget.attempts)
    action = "retry" if budget.action == "retry" else "fallback"
    return SpeechDecision(action, None, budget.target, budget.attempts)
