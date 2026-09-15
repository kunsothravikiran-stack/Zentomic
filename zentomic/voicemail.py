"""Offline admission of hosted Record action callback results."""

import re
from dataclasses import dataclass
from typing import Literal


MAX_VOICEMAIL_SECONDS = 600
_DURATION = re.compile(r"0|[1-9][0-9]{0,2}")


@dataclass(frozen=True)
class VoicemailDecision:
    """Sanitized call-flow decision without recording URLs or caller audio."""

    action: Literal["continue", "hangup"]
    duration_seconds: int
    ended_by: Literal["silence-or-limit", "finish-key", "hangup"]


def _validate_voicemail_result_configuration(
    *, max_length: int, finish_on_key: str,
) -> None:
    """Validate the trusted policy that produced the corresponding Record."""
    if type(max_length) is not int or not 2 <= max_length <= MAX_VOICEMAIL_SECONDS:
        raise ValueError("max_length must be an integer from 2 to 600 seconds")
    if (not isinstance(finish_on_key, str) or len(finish_on_key) != 1
            or finish_on_key not in "0123456789*#"):
        raise ValueError("finish_on_key must be one DTMF character")


def _resolve_validated_voicemail_result(
    recording_duration: str | None, digits: str | None, *,
    max_length: int, finish_on_key: str,
) -> VoicemailDecision:
    """Admit callback fields after trusted policy validation."""
    if (not isinstance(recording_duration, str)
            or len(recording_duration) > 3
            or not _DURATION.fullmatch(recording_duration)):
        raise ValueError("recording duration must be a canonical nonnegative integer")
    duration_seconds = int(recording_duration)
    if duration_seconds > max_length:
        raise ValueError("recording duration exceeds the configured maximum")

    if digits is None or digits == "":
        return VoicemailDecision("continue", duration_seconds, "silence-or-limit")
    if digits == finish_on_key:
        return VoicemailDecision("continue", duration_seconds, "finish-key")
    if digits == "hangup":
        return VoicemailDecision("hangup", duration_seconds, "hangup")
    raise ValueError("unsupported voicemail completion input")


def resolve_voicemail_result(
    recording_duration: str | None, digits: str | None, *,
    max_length: int = 120, finish_on_key: str = "#",
) -> VoicemailDecision:
    """Return a bounded decision from an authenticated Record action callback.

    ``recording_duration`` and ``digits`` correspond to Twilio's Record action
    fields. Missing/blank Digits means silence or the configured length stopped
    recording; the configured finish key allows continuation; ``hangup`` keeps
    a disconnected caller terminal. Duration is an initial callback value, not
    proof that audio is available or its final post-trim duration.

    Call this only after authenticating and binding the callback. The helper
    deliberately accepts neither RecordingUrl nor audio, and performs no I/O,
    persistence, retrieval, acknowledgement, or recording-status handling.
    """
    _validate_voicemail_result_configuration(
        max_length=max_length, finish_on_key=finish_on_key,
    )
    return _resolve_validated_voicemail_result(
        recording_duration, digits,
        max_length=max_length, finish_on_key=finish_on_key,
    )
