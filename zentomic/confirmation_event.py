"""Authenticated pending-intent confirmation without persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.gather import GatherDecision
from zentomic.intent import resolve_intent_confirmation


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
    if not isinstance(routes, Mapping):
        raise ValueError("routes must be a mapping")
    menu = dict(routes)
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid, expected_call_sid=expected_call_sid,
    )
    return resolve_intent_confirmation(
        fields.get("Digits"), pending_intent, menu,
        fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
