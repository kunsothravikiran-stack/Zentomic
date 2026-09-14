"""Authenticated pending-intent confirmation without persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.callback_claim_store import (
    CallbackClaimer,
    validate_and_claim_call_event,
)
from zentomic.gather import GatherDecision
from zentomic.intent import (
    _resolve_validated_intent_confirmation,
    _snapshot_intent_routes,
    _validate_intent_confirmation_configuration,
    resolve_intent_confirmation,
)


def resolve_confirmation_event(
    event: Mapping[str, Any], *, pending_intent: str | None,
    routes: Mapping[str, str], public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    fallback_target: str, attempts: int, max_attempts: int = 3,
    hangup_digit: str | None = None,
) -> GatherDecision:
    """Authenticate a callback, then propose bounded pending-intent confirmation.

    All keyword arguments must come from trusted workspace/session state for
    this exact confirmation step, never callback fields. Snapshot routes before
    invoking the validator. Only authenticated Digits enters the existing
    confirmation policy; missing input is silence, not consent. Configuration
    validation follows the existing policy after authentication.

    Account/call binding does not bind a pending intent to the current step or
    prevent replay. Before acting, enforce deadlines and transition budgets,
    atomically claim the unchanged confirmation step and persist attempts.
    Reload and revalidate on conflicts; never apply a late confirmation to a
    changed intent. Authorize all destinations within the workspace separately.
    This helper does not load state, classify, render TwiML, persist, acknowledge
    callbacks, or place calls. A route decision alone does not authorize dialing.
    """
    menu = _snapshot_intent_routes(routes)
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid, expected_call_sid=expected_call_sid,
    )
    return resolve_intent_confirmation(
        fields.get("Digits"), pending_intent, menu,
        fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts, hangup_digit=hangup_digit,
    )


def resolve_claimed_confirmation_event(
    event: Mapping[str, Any], *, pending_intent: str | None,
    routes: Mapping[str, str], public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    workspace_id: str, step_id: str, claimer: CallbackClaimer,
    fallback_target: str, attempts: int, max_attempts: int = 3,
    hangup_digit: str | None = None,
) -> GatherDecision:
    """Validate trusted state, authenticate, claim, then confirm one intent.

    Pending intent, routes, retry state and claim key must come from one trusted
    session snapshot. Every policy setting is privately copied and validated
    before authentication and claiming. Rejected callbacks therefore leave the
    step unclaimed, while authenticated replays fail before caller input reaches
    confirmation policy. Signed fields cannot replace trusted state.

    Production adapters still need an authorized current-step and unchanged-
    intent check plus a durable conditional write, transactionally combined
    with related session state when required. This helper performs no I/O,
    persistence, classification, rendering, acknowledgement or dialing.
    """
    confirmation_menu, known = _validate_intent_confirmation_configuration(
        pending_intent, routes, fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts,
        hangup_digit=hangup_digit,
    )
    fields = validate_and_claim_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid, workspace_id=workspace_id,
        step_id=step_id, claimer=claimer,
    )
    return _resolve_validated_intent_confirmation(
        fields.get("Digits"), confirmation_menu, known=known,
        fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
