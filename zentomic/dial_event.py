"""Authenticated Number dial-result composition without persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.callback_claim_store import (
    CallbackClaimer,
    _commit_callback_claim,
    _prepare_callback_claim,
)
from zentomic.dial import (
    DialDecision,
    _resolve_validated_dial_result,
    _validate_dial_result_configuration,
    resolve_dial_result,
)
from zentomic.routing import _valid_target


def resolve_dial_result_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str, expected_dial_call_sid: str,
    fallback_target: str, fallback_used: bool = False,
) -> DialDecision:
    """Authenticate a Number action callback and propose a forwarding decision.

    All keyword arguments must come from trusted workspace configuration and
    session state. expected_call_sid is the parent; expected_dial_call_sid is
    the already known child for the current dial step, never learned from this
    request. Missing or mismatched child identity fails before outcome policy.
    This strict boundary cannot be used until trusted child state is available.

    Read only DialCallStatus for policy, not parent CallStatus. Even a consumed
    fallback requires valid authenticated input. Return no raw request fields.
    This does not authorize targets, load state, deduplicate, persist, render,
    acknowledge, or dial. Atomically claim the current step and mark fallback
    used before acting; reload and revalidate on a concurrent state change.
    """
    if not _valid_target(expected_dial_call_sid):
        raise ValueError(
            "expected_dial_call_sid must be a nonblank UTF-8 string of at most 256 bytes"
        )
    if not _valid_target(fallback_target):
        raise ValueError("fallback_target must be a nonblank UTF-8 string of at most 256 bytes")
    if type(fallback_used) is not bool:
        raise ValueError("fallback_used must be a boolean")
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid, expected_call_sid=expected_call_sid,
    )
    if fields.get("DialCallSid") != expected_dial_call_sid:
        raise ValueError("webhook does not match the expected dial step")
    return resolve_dial_result(
        fields.get("DialCallStatus"), fallback_target=fallback_target,
        fallback_used=fallback_used,
    )


def resolve_claimed_dial_result_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str, expected_dial_call_sid: str,
    workspace_id: str, step_id: str, claimer: CallbackClaimer,
    fallback_target: str, fallback_used: bool = False,
) -> DialDecision:
    """Validate trusted state, bind the child leg, claim, then resolve once.

    Parent and child call identifiers, workspace, step and fallback state must
    come from one trusted session snapshot. Configuration and the claim boundary
    are validated before authentication. A rejected signature, wrong parent or
    stale child callback leaves the step unclaimed. The exact child identity is
    checked before claiming, while replays fail before outcome policy runs.

    Production adapters still need an authorized current-step and child-leg
    check plus a durable conditional write, transactionally combined with
    fallback state when required. This helper performs no I/O, persistence,
    rendering, callback acknowledgement, cleanup, or dialing.
    """
    if not _valid_target(expected_dial_call_sid):
        raise ValueError(
            "expected_dial_call_sid must be a nonblank UTF-8 string of at most 256 bytes"
        )
    _validate_dial_result_configuration(
        fallback_target=fallback_target, fallback_used=fallback_used,
    )
    prepared = _prepare_callback_claim(
        workspace_id, expected_call_sid, step_id, claimer,
    )
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
    )
    if fields.get("DialCallSid") != expected_dial_call_sid:
        raise ValueError("webhook does not match the expected dial step")
    _commit_callback_claim(prepared)
    return _resolve_validated_dial_result(
        fields.get("DialCallStatus"), fallback_target=fallback_target,
        fallback_used=fallback_used,
    )
