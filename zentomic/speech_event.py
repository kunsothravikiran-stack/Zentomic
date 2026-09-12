"""Authenticated speech admission without classification, persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.speech import SpeechDecision, resolve_speech_gather


def resolve_speech_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    fallback_target: str, attempts: int, max_attempts: int = 3,
) -> SpeechDecision:
    """Authenticate a speech callback, then propose bounded text admission.

    All keyword arguments come from trusted workspace/session configuration,
    never callback fields. Only authenticated SpeechResult enters the existing
    speech policy; missing input is silence. Policy configuration is validated
    after authentication. Accepted text remains untrusted and is not logged.

    Account/call binding does not establish the current collection step or
    prevent replay. Before classification or reprompting, enforce deadlines
    and transition budgets, atomically claim the step and persist attempts.
    Reload and revalidate on conflicts. A classify decision is not permission
    to spend: apply model/token/cost limits and authorize workspace access.
    This helper does not call a model, confirm an intent, render TwiML, persist,
    acknowledge callbacks, place calls, or make a routing decision.
    """
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid, expected_call_sid=expected_call_sid,
    )
    return resolve_speech_gather(
        fields.get("SpeechResult"), fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts,
    )
