"""Authenticated keypad collection composition without persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.gather import GatherDecision, resolve_gather


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
    if not isinstance(routes, Mapping):
        raise ValueError("routes must be a mapping")
    menu = dict(routes)
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid, expected_call_sid=expected_call_sid,
    )
    return resolve_gather(
        fields.get("Digits"), menu, fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
