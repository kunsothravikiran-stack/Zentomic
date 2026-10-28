"""Strict, offline admission of voicemail recording-status callbacks."""

import re
from dataclasses import dataclass
from typing import Literal

from zentomic.voicemail import _validate_voicemail_max_length


_DURATION = re.compile(r"0|[1-9][0-9]{0,2}")
_RECORDING_SID = re.compile(r"RE[0-9A-Fa-f]{32}")


@dataclass(frozen=True)
class VoicemailRecordingStatus:
    """Sanitized recording availability without a media URL or caller audio."""

    availability: Literal["available", "unavailable"]
    recording_sid: str
    duration_seconds: int | None


def _validate_recording_sid(recording_sid: str | None) -> str:
    """Return one canonical provider recording identifier or fail closed."""
    if (not isinstance(recording_sid, str)
            or len(recording_sid) > 34
            or not _RECORDING_SID.fullmatch(recording_sid)):
        raise ValueError("recording SID must use Twilio RE identifier syntax")
    return recording_sid


def _resolve_validated_voicemail_recording_status(
    recording_sid: str | None, recording_status: str | None,
    recording_duration: str | None, recording_channels: str | None,
    recording_source: str | None, *, max_length: int,
) -> VoicemailRecordingStatus:
    """Admit documented Record status fields after trusted policy validation."""
    recording_sid = _validate_recording_sid(recording_sid)
    if recording_status not in ("completed", "failed"):
        raise ValueError("unsupported recording status")
    if recording_channels != "1":
        raise ValueError("Record callbacks must report exactly one channel")
    if recording_source != "RecordVerb":
        raise ValueError("recording source must be RecordVerb")
    if (not isinstance(recording_duration, str)
            or len(recording_duration) > 3
            or not _DURATION.fullmatch(recording_duration)):
        raise ValueError("recording duration must be a canonical nonnegative integer")
    duration_seconds = int(recording_duration)
    if duration_seconds > max_length:
        raise ValueError("recording duration exceeds the configured maximum")

    if recording_status == "completed":
        return VoicemailRecordingStatus(
            "available", recording_sid, duration_seconds,
        )
    return VoicemailRecordingStatus("unavailable", recording_sid, None)


def resolve_voicemail_recording_status(
    recording_sid: str | None, recording_status: str | None,
    recording_duration: str | None, recording_channels: str | None,
    recording_source: str | None, *, max_length: int = 120,
) -> VoicemailRecordingStatus:
    """Return a bounded decision from an authenticated recording-status callback.

    The accepted fields match the Record verb's completed/failed status callback
    contract. A failed recording never exposes its reported duration as usable
    media. Call this only after authenticating and binding the callback.

    This helper deliberately accepts neither RecordingUrl nor audio, and performs
    no I/O, persistence, retrieval, acknowledgement, or recording operation.
    """
    _validate_voicemail_max_length(max_length)
    return _resolve_validated_voicemail_recording_status(
        recording_sid, recording_status, recording_duration,
        recording_channels, recording_source, max_length=max_length,
    )
