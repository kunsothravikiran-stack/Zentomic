"""Authenticated Number dial-result composition without persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.dial import DialDecision, resolve_dial_result
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
