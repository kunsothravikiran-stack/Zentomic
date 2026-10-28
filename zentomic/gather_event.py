"""Authenticated keypad collection composition without persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.callback_claim_store import (
    CallbackClaimer,
    validate_and_claim_call_event,
)
from zentomic.gather import (
    GatherDecision,
    _resolve_validated_gather,
    _validate_gather_configuration,
    resolve_gather,
)
from zentomic.routing import _snapshot_dtmf_routes


def resolve_gather_event(
    event: Mapping[str, Any], routes: Mapping[str, str], *,
    public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    fallback_target: str, attempts: int, max_attempts: int = 3,
    hangup_digit: str | None = None,
) -> GatherDecision:
    """Authenticate a keypad callback, then propose one collection decision.

    Routes and all keyword arguments come from trusted workspace/session
    state, never request fields. Snapshot routes before invoking the validator.
    Only authenticated Digits enters policy; missing input is silence. The
    existing gather policy validates configuration after authentication.

    Account/call binding is not collection-step binding or replay protection.
    Before acting, enforce the current step, deadline and transition budget;
    atomically claim that step and persist the returned attempt count. Reload
    and revalidate on conflicts. This helper does not load state, authorize
    destinations, render TwiML, persist, acknowledge callbacks, or place calls.
    """
    menu = _snapshot_dtmf_routes(routes)
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid, expected_call_sid=expected_call_sid,
    )
    return resolve_gather(
        fields.get("Digits"), menu, fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts, hangup_digit=hangup_digit,
    )


def resolve_claimed_gather_event(
    event: Mapping[str, Any], routes: Mapping[str, str], *,
    public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    workspace_id: str, step_id: str, claimer: CallbackClaimer,
    fallback_target: str, attempts: int, max_attempts: int = 3,
    hangup_digit: str | None = None,
) -> GatherDecision:
    """Validate trusted state, authenticate, claim, then resolve one keypad step.

    The workspace, call, step, retry count, routes and destinations must come
    from the same trusted session snapshot. Every policy setting is validated
    before authentication and before the claim, so bad adapter configuration
    cannot consume a valid step. Invalid or identity-mismatched callbacks also
    do not consume it. A replay raises ``CallbackReplayError`` before keypad
    policy runs; claim storage failures propagate and fail closed.

    The claim deliberately precedes policy evaluation and any caller-owned
    effect. Production adapters still need an authorized current-step check and
    a durable conditional write, combined transactionally with related session
    state where required. This helper performs no I/O, persistence, rendering,
    callback acknowledgement, destination authorization, or telephony action.
    """
    menu = _validate_gather_configuration(
        routes, fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
    fields = validate_and_claim_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid, workspace_id=workspace_id,
        step_id=step_id, claimer=claimer,
    )
    return _resolve_validated_gather(
        fields.get("Digits"), menu, fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
