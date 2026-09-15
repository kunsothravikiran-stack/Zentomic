"""Authenticated voicemail recording-status composition without I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.callback_claim_store import (
    CallbackClaimer,
    _commit_callback_claim,
    _prepare_callback_claim,
)
from zentomic.voicemail import _validate_voicemail_max_length
from zentomic.voicemail_status import (
    VoicemailRecordingStatus,
    _resolve_validated_voicemail_recording_status,
    _validate_recording_sid,
)
from zentomic.voicemail_status_store import (
    VoicemailStatusStore,
    _key as _status_key,
    _validate_status,
)


def _resolve_fields(
    fields: Mapping[str, str], *, max_length: int,
    expected_recording_sid: str | None,
) -> VoicemailRecordingStatus:
    result = _resolve_validated_voicemail_recording_status(
        fields.get("RecordingSid"), fields.get("RecordingStatus"),
        fields.get("RecordingDuration"), fields.get("RecordingChannels"),
        fields.get("RecordingSource"), max_length=max_length,
    )
    if (expected_recording_sid is not None
            and result.recording_sid != expected_recording_sid):
        raise ValueError("recording SID does not match the expected recording")
    return result


def resolve_voicemail_recording_status_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str, max_length: int = 120,
    expected_recording_sid: str | None = None,
) -> VoicemailRecordingStatus:
    """Authenticate, bind, and sanitize one Record status callback.

    Expected identifiers and max_length must come from trusted session state.
    When supplied, expected_recording_sid also binds the callback to the exact
    recording already associated with that state. RecordingUrl and unrelated
    fields are ignored and never returned. This helper performs no persistence,
    retrieval, acknowledgement, or I/O.
    """
    _validate_voicemail_max_length(max_length)
    if expected_recording_sid is not None:
        _validate_recording_sid(expected_recording_sid)
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
    )
    return _resolve_fields(
        fields, max_length=max_length,
        expected_recording_sid=expected_recording_sid,
    )


def resolve_claimed_voicemail_recording_status_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str, workspace_id: str,
    status_step_id: str, claimer: CallbackClaimer, max_length: int = 120,
    expected_recording_sid: str | None = None,
) -> VoicemailRecordingStatus:
    """Validate, authenticate, admit, and claim one recording-status callback.

    Use a status_step_id distinct from the Record action callback's step. The
    callback is admitted before its trusted claim is consumed, so malformed or
    rejected deliveries cannot block a later valid delivery. Production needs
    a durable, workspace-authorized claim. When expected_recording_sid is
    available from trusted state, it is checked before the claim is consumed.
    This helper performs no I/O.
    """
    _validate_voicemail_max_length(max_length)
    if expected_recording_sid is not None:
        _validate_recording_sid(expected_recording_sid)
    prepared = _prepare_callback_claim(
        workspace_id, expected_call_sid, status_step_id, claimer,
    )
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
    )
    result = _resolve_fields(
        fields, max_length=max_length,
        expected_recording_sid=expected_recording_sid,
    )
    _commit_callback_claim(prepared)
    return result


def record_voicemail_recording_status_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    expected_recording_sid: str, workspace_id: str,
    store: VoicemailStatusStore, max_length: int = 120,
) -> VoicemailRecordingStatus:
    """Authenticate, bind, sanitize, and immutably record one final status.

    Every identifier is trusted workspace/session state. The exact storage key
    and adapter shape are validated before request authentication. The recording
    SID is always bound to expected_recording_sid, so signed callback fields
    cannot select another recording key. Invalid callbacks never reach storage.

    The adapter must make exact redelivery idempotent and reject contradictory
    final metadata. Its returned value is validated against the admitted result.
    Storage errors propagate and fail closed. This helper performs no media
    retrieval, callback acknowledgement, replay claiming, or provider I/O.
    """
    _validate_voicemail_max_length(max_length)
    _validate_recording_sid(expected_recording_sid)
    record = getattr(store, "record", None)
    if not callable(record):
        raise ValueError("store must provide a trusted callable record method")
    _status_key(workspace_id, expected_call_sid, expected_recording_sid)
    result = resolve_voicemail_recording_status_event(
        event,
        public_url=public_url,
        validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
        max_length=max_length,
        expected_recording_sid=expected_recording_sid,
    )
    saved = _validate_status(record(workspace_id, expected_call_sid, result))
    if saved != result:
        raise ValueError("store returned an unexpected VoicemailRecordingStatus")
    return saved
