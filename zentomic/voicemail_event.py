"""Authenticated voicemail action composition without persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.callback_claim_store import (
    CallbackClaimer,
    _commit_callback_claim,
    _prepare_callback_claim,
)
from zentomic.voicemail import (
    VoicemailDecision,
    _resolve_validated_voicemail_result,
    _validate_voicemail_result_configuration,
)


def resolve_voicemail_result_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    max_length: int = 120, finish_on_key: str = "#",
) -> VoicemailDecision:
    """Authenticate a Record action callback and return sanitized call flow.

    The expected identifiers and Record policy must come from trusted session
    state. Callback fields cannot change the duration bound or finish key.
    RecordingUrl and all unrelated fields are ignored and never returned.

    Account/call binding is not current-step authorization or replay defense.
    Before continuing call flow, enforce the current voicemail step and claim it
    atomically. Use the separate recording-status callback for availability and
    final duration. This helper performs no persistence, retrieval, rendering,
    acknowledgement, telephony action, logging, or recording-status handling.
    """
    _validate_voicemail_result_configuration(
        max_length=max_length, finish_on_key=finish_on_key,
    )
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
    )
    return _resolve_validated_voicemail_result(
        fields.get("RecordingDuration"), fields.get("Digits"),
        max_length=max_length, finish_on_key=finish_on_key,
    )


def resolve_claimed_voicemail_result_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    workspace_id: str, step_id: str, claimer: CallbackClaimer,
    max_length: int = 120, finish_on_key: str = "#",
) -> VoicemailDecision:
    """Validate, authenticate, admit and claim one Record action callback.

    Trusted policy and the claim boundary are validated before authentication.
    Authentication and result admission finish before the claim, so rejected or
    malformed callbacks cannot consume the current step. A successful callback
    is claimed before its sanitized decision is returned; replays fail closed.

    Production needs a durable authorized current-step claim, transactionally
    combined with session changes where required. The recording-status callback
    needs its own independently authenticated and deduplicated lifecycle. This
    helper performs no I/O and never returns RecordingUrl or caller audio.
    """
    _validate_voicemail_result_configuration(
        max_length=max_length, finish_on_key=finish_on_key,
    )
    prepared = _prepare_callback_claim(
        workspace_id, expected_call_sid, step_id, claimer,
    )
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
    )
    decision = _resolve_validated_voicemail_result(
        fields.get("RecordingDuration"), fields.get("Digits"),
        max_length=max_length, finish_on_key=finish_on_key,
    )
    _commit_callback_claim(prepared)
    return decision
