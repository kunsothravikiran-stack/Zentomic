"""Authenticated, claimed business-hours decisions without clock reads or I/O."""

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from zentomic.authentication import SignatureValidator
from zentomic.business_hours import resolve_business_hours_target_at
from zentomic.callback_claim_store import CallbackClaimer, validate_and_claim_call_event


def resolve_claimed_business_hours_event(
    event: Mapping[str, Any], hours: Mapping[int, Sequence[tuple[int, int]]], *,
    instant: datetime, timezone_name: str, open_target: str, closed_target: str,
    public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    workspace_id: str, step_id: str, claimer: CallbackClaimer,
    date_overrides: Mapping[date, Sequence[tuple[int, int]]] | None = None,
) -> str:
    """Return one as-of-time target only after authenticating and claiming a step.

    All settings, including the fresh aware instant, authorized workspace zone,
    schedule, date exceptions and claim key, come from trusted session state.
    No callback time, timezone, target or step field influences the decision.
    Resolve and validate the entire policy before invoking dependencies, so bad
    configuration cannot consume a valid step and neither the validator nor
    claimer can rewrite the selected target by mutating live configuration.

    Authentication and account/call binding precede the atomic claim. Replays
    raise CallbackReplayError; claim failures propagate without returning a
    target. No policy is re-read after consuming the claim. This is an as-of
    decision, not a clock-freshness check: authentication/storage latency may
    cross an opening boundary. Callers must enforce freshness, current step,
    terminal state and budgets, authorize the destination, and transactionally
    persist related session state before acting. A consumed claim is not proof
    that a downstream effect succeeded. No clock, SDK, storage or telephony is
    constructed here; only explicitly injected dependencies are invoked.
    """
    target = resolve_business_hours_target_at(
        hours, instant=instant, timezone_name=timezone_name,
        open_target=open_target, closed_target=closed_target,
        date_overrides=date_overrides,
    )
    validate_and_claim_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid, workspace_id=workspace_id,
        step_id=step_id, claimer=claimer,
    )
    return target
