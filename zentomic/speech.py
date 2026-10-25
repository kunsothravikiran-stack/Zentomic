"""Offline, bounded admission of speech results to a future intent classifier."""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from zentomic.gather import GatherDecision, resolve_gather


MAX_TRANSCRIPT_CHARS = 2000
MAX_TRANSCRIPT_UTF8_BYTES = MAX_TRANSCRIPT_CHARS * 4
MAX_TRANSCRIPT_WORDS = MAX_TRANSCRIPT_CHARS // 2
MAX_TRANSCRIPT_LINES = MAX_TRANSCRIPT_CHARS
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


def _validate_transcript_policy(
    *, min_transcript_characters: int = 1,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    min_transcript_utf8_bytes: int = 1,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
    min_transcript_words: int = 1,
    max_transcript_words: int = MAX_TRANSCRIPT_WORDS,
    min_transcript_lines: int = 1,
    max_transcript_lines: int = MAX_TRANSCRIPT_LINES,
    require_transcript_alphanumeric: bool = False,
) -> None:
    """Validate trusted transcript bounds, including cross-limit feasibility."""
    if (type(min_transcript_characters) is not int
            or not 1 <= min_transcript_characters <= MAX_TRANSCRIPT_CHARS):
        raise ValueError(
            "min_transcript_characters must be an integer from 1 to 2000"
        )
    if (type(max_transcript_characters) is not int
            or not 1 <= max_transcript_characters <= MAX_TRANSCRIPT_CHARS):
        raise ValueError(
            "max_transcript_characters must be an integer from 1 to 2000"
        )
    if min_transcript_characters > max_transcript_characters:
        raise ValueError(
            "min_transcript_characters must not exceed max_transcript_characters"
        )
    if (type(min_transcript_utf8_bytes) is not int
            or not 1 <= min_transcript_utf8_bytes <= MAX_TRANSCRIPT_UTF8_BYTES):
        raise ValueError(
            "min_transcript_utf8_bytes must be an integer from 1 to 8000"
        )
    if (type(max_transcript_utf8_bytes) is not int
            or not 1 <= max_transcript_utf8_bytes <= MAX_TRANSCRIPT_UTF8_BYTES):
        raise ValueError(
            "max_transcript_utf8_bytes must be an integer from 1 to 8000"
        )
    if min_transcript_utf8_bytes > max_transcript_utf8_bytes:
        raise ValueError(
            "min_transcript_utf8_bytes must not exceed max_transcript_utf8_bytes"
        )
    if (type(min_transcript_words) is not int
            or not 1 <= min_transcript_words <= MAX_TRANSCRIPT_WORDS):
        raise ValueError(
            "min_transcript_words must be an integer from 1 to 1000"
        )
    if (type(max_transcript_words) is not int
            or not 1 <= max_transcript_words <= MAX_TRANSCRIPT_WORDS):
        raise ValueError(
            "max_transcript_words must be an integer from 1 to 1000"
        )
    if min_transcript_words > max_transcript_words:
        raise ValueError(
            "min_transcript_words must not exceed max_transcript_words"
        )
    if (type(min_transcript_lines) is not int
            or not 1 <= min_transcript_lines <= MAX_TRANSCRIPT_LINES):
        raise ValueError(
            "min_transcript_lines must be an integer from 1 to 2000"
        )
    if (type(max_transcript_lines) is not int
            or not 1 <= max_transcript_lines <= MAX_TRANSCRIPT_LINES):
        raise ValueError(
            "max_transcript_lines must be an integer from 1 to 2000"
        )
    if min_transcript_lines > max_transcript_lines:
        raise ValueError(
            "min_transcript_lines must not exceed max_transcript_lines"
        )
    if type(require_transcript_alphanumeric) is not bool:
        raise ValueError("require_transcript_alphanumeric must be a boolean")

    # Every character occupies at least one UTF-8 byte. Each nonempty word
    # needs at least one character, with at least one whitespace character
    # between adjacent words. Reject trusted policies that cannot admit any
    # transcript instead of silently spending every collection attempt.
    if min_transcript_characters > max_transcript_utf8_bytes:
        raise ValueError(
            "min_transcript_characters must not exceed max_transcript_utf8_bytes"
        )
    minimum_word_characters = min_transcript_words * 2 - 1
    if minimum_word_characters > max_transcript_characters:
        raise ValueError(
            "min_transcript_words cannot fit within max_transcript_characters"
        )
    if minimum_word_characters > max_transcript_utf8_bytes:
        raise ValueError(
            "min_transcript_words cannot fit within max_transcript_utf8_bytes"
        )
    if min_transcript_lines > max_transcript_characters:
        raise ValueError(
            "min_transcript_lines cannot fit within max_transcript_characters"
        )
    if min_transcript_lines > max_transcript_utf8_bytes:
        raise ValueError(
            "min_transcript_lines cannot fit within max_transcript_utf8_bytes"
        )
    minimum_separators = max(min_transcript_words - 1, min_transcript_lines - 1)
    minimum_structural_characters = min_transcript_words + minimum_separators
    if min_transcript_words == 1 and min_transcript_lines > 1:
        leading_line_characters = (
            min_transcript_characters + min_transcript_lines - 1
        )
        minimum_policy_characters = leading_line_characters
        if max_transcript_words >= 2:
            internal_line_characters = max(
                min_transcript_characters, min_transcript_lines + 1,
            )
            minimum_policy_characters = min(
                leading_line_characters, internal_line_characters,
            )
    else:
        minimum_policy_characters = max(
            min_transcript_characters, minimum_structural_characters,
        )
    if minimum_policy_characters > max_transcript_characters:
        raise ValueError(
            "minimum transcript words and lines cannot fit within "
            "max_transcript_characters"
        )
    if minimum_policy_characters > max_transcript_utf8_bytes:
        raise ValueError(
            "minimum transcript words and lines cannot fit within "
            "max_transcript_utf8_bytes"
        )
    # A Unicode scalar occupies at most four UTF-8 bytes. Required word and
    # line separators can share characters, and Python-recognized Unicode
    # whitespace occupies at most three bytes per character.
    if max_transcript_words == 1 and min_transcript_lines > 1:
        # With only one word, required line separators must sit outside the
        # trimmed content and cannot help meet its byte minimum.
        maximum_content_bytes = (
            max_transcript_characters - (min_transcript_lines - 1)
        ) * 4
    else:
        maximum_content_bytes = (
            max_transcript_characters * 4 - minimum_separators
        )
    if min_transcript_utf8_bytes > maximum_content_bytes:
        raise ValueError(
            "min_transcript_utf8_bytes cannot fit within max_transcript_characters "
            "and min_transcript_words"
        )


def _validate_speech_configuration(
    *, fallback_target: str, attempts: int, max_attempts: int,
    minimum_confidence_percent: int | None = None,
    require_confidence: bool = False,
    min_transcript_characters: int = 1,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    min_transcript_utf8_bytes: int = 1,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
    min_transcript_words: int = 1,
    max_transcript_words: int = MAX_TRANSCRIPT_WORDS,
    min_transcript_lines: int = 1,
    max_transcript_lines: int = MAX_TRANSCRIPT_LINES,
    require_transcript_alphanumeric: bool = False,
) -> GatherDecision:
    """Return the collection budget after validating trusted speech policy."""
    if (minimum_confidence_percent is not None
            and (type(minimum_confidence_percent) is not int
                 or not 0 <= minimum_confidence_percent <= 100)):
        raise ValueError("minimum_confidence_percent must be an integer from 0 to 100 or None")
    if type(require_confidence) is not bool:
        raise ValueError("require_confidence must be a boolean")
    _validate_transcript_policy(
        min_transcript_characters=min_transcript_characters,
        max_transcript_characters=max_transcript_characters,
        min_transcript_utf8_bytes=min_transcript_utf8_bytes,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
        min_transcript_words=min_transcript_words,
        max_transcript_words=max_transcript_words,
        min_transcript_lines=min_transcript_lines,
        max_transcript_lines=max_transcript_lines,
        require_transcript_alphanumeric=require_transcript_alphanumeric,
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
    min_transcript_characters: int = 1,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    min_transcript_utf8_bytes: int = 1,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
    min_transcript_words: int = 1,
    max_transcript_words: int = MAX_TRANSCRIPT_WORDS,
    min_transcript_lines: int = 1,
    max_transcript_lines: int = MAX_TRANSCRIPT_LINES,
    require_transcript_alphanumeric: bool = False,
) -> SpeechDecision:
    """Admit speech after the caller validated and snapshotted its budget."""
    if (attempts < max_attempts and isinstance(speech, str)
            and len(speech) <= max_transcript_characters):
        stripped_speech = speech.strip()
        admissible = (
            len(stripped_speech) >= min_transcript_characters
            and (not require_transcript_alphanumeric
                 or any(character.isalnum() for character in stripped_speech))
            and _confidence_meets_minimum(
                confidence, minimum_confidence_percent,
                require_confidence=require_confidence,
            )
        )
    else:
        admissible = False
    if admissible:
        try:
            encoded_speech = speech.encode("utf-8")
        except UnicodeEncodeError:
            pass  # Malformed text follows the same finite retry budget.
        else:
            # Split at most one segment beyond the trusted limit. This bounds
            # temporary work while treating all Unicode whitespace consistently.
            word_count = len(speech.split(None, max_transcript_words))
            line_count = len(speech.splitlines()) or 1
            stripped_byte_count = len(stripped_speech.encode("utf-8"))
            if (min_transcript_utf8_bytes <= stripped_byte_count
                    and len(encoded_speech) <= max_transcript_utf8_bytes
                    and min_transcript_words <= word_count <= max_transcript_words
                    and min_transcript_lines <= line_count <= max_transcript_lines):
                return SpeechDecision("classify", speech, None, budget.attempts)
    action = "retry" if budget.action == "retry" else "fallback"
    return SpeechDecision(action, None, budget.target, budget.attempts)


def resolve_speech_gather(
    speech: str | None, *, fallback_target: str, attempts: int, max_attempts: int = 3,
    confidence: str | None = None, minimum_confidence_percent: int | None = None,
    require_confidence: bool = False,
    min_transcript_characters: int = 1,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    min_transcript_utf8_bytes: int = 1,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
    min_transcript_words: int = 1,
    max_transcript_words: int = MAX_TRANSCRIPT_WORDS,
    min_transcript_lines: int = 1,
    max_transcript_lines: int = MAX_TRANSCRIPT_LINES,
    require_transcript_alphanumeric: bool = False,
) -> SpeechDecision:
    """Admit bounded nonblank text, or retry silence within a trusted budget.

    Reuse the keypad collection budget and configuration validation. Missing,
    malformed, blank, oversized, or explicitly low-confidence results consume
    one attempt but never reach a classifier. A configured confidence threshold
    is an integer percentage from 0 through 100. Missing provider confidence
    remains admissible because the field is not guaranteed, unless trusted
    configuration explicitly requires it. Under either confidence policy, a
    present score must be a bounded decimal from 0 through 1. The trusted
    minimum transcript length is an integer from 1 through 2000 and cannot
    exceed the configured character ceiling. It counts characters after
    trimming surrounding Unicode whitespace. The trusted minimum UTF-8 byte
    count is an integer from 1 through 8000 and measures the same trimmed text.
    The trusted minimum word count is
    an integer from 1 through 1000 and cannot exceed the configured word ceiling.
    The trusted minimum line count is an integer from 1 through 2000 and cannot
    exceed the configured line ceiling. Maximum transcript limits are integers
    from 1 through the fixed
    2000-character, 8000-byte, 1000-word and 2000-line ceilings. Maximum
    character and byte limits include surrounding whitespace; words are
    nonempty Unicode-whitespace-delimited segments. Lines use Python's Unicode
    line-boundary rules. Byte length is measured from strict UTF-8 encoding;
    accepted text is returned unchanged. Optional alphanumeric policy requires
    at least one Unicode letter or number, rejecting punctuation-, symbol-, and
    mark-only recognition results without imposing an ASCII-only language rule.
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
        min_transcript_characters=min_transcript_characters,
        max_transcript_characters=max_transcript_characters,
        min_transcript_utf8_bytes=min_transcript_utf8_bytes,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
        min_transcript_words=min_transcript_words,
        max_transcript_words=max_transcript_words,
        min_transcript_lines=min_transcript_lines,
        max_transcript_lines=max_transcript_lines,
        require_transcript_alphanumeric=require_transcript_alphanumeric,
    )
    return _resolve_validated_speech(
        speech, budget=budget, attempts=attempts, max_attempts=max_attempts,
        confidence=confidence,
        minimum_confidence_percent=minimum_confidence_percent,
        require_confidence=require_confidence,
        min_transcript_characters=min_transcript_characters,
        max_transcript_characters=max_transcript_characters,
        min_transcript_utf8_bytes=min_transcript_utf8_bytes,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
        min_transcript_words=min_transcript_words,
        max_transcript_words=max_transcript_words,
        min_transcript_lines=min_transcript_lines,
        max_transcript_lines=max_transcript_lines,
        require_transcript_alphanumeric=require_transcript_alphanumeric,
    )
