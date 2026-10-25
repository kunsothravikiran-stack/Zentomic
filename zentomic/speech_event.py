"""Authenticated speech admission without classification, persistence or I/O."""

from collections.abc import Mapping
from typing import Any

from zentomic.authentication import SignatureValidator, validate_call_event
from zentomic.callback_claim_store import (
    CallbackClaimer,
    validate_and_claim_call_event,
)
from zentomic.speech import (
    MAX_TRANSCRIPT_CHARS,
    MAX_TRANSCRIPT_UTF8_BYTES,
    MAX_TRANSCRIPT_WORDS,
    SpeechDecision,
    _resolve_validated_speech,
    _validate_speech_configuration,
    resolve_speech_gather,
)


def resolve_speech_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    fallback_target: str, attempts: int, max_attempts: int = 3,
    minimum_confidence_percent: int | None = None,
    require_confidence: bool = False,
    min_transcript_characters: int = 1,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    min_transcript_utf8_bytes: int = 1,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
    min_transcript_words: int = 1,
    max_transcript_words: int = MAX_TRANSCRIPT_WORDS,
) -> SpeechDecision:
    """Authenticate a speech callback, then propose bounded text admission.

    All keyword arguments come from trusted workspace/session configuration,
    never callback fields. Only authenticated SpeechResult enters the existing
    speech policy; missing input is silence. An optional trusted confidence
    threshold rejects an explicitly lower or malformed provider score, while a
    missing score remains admissible because the provider does not guarantee it,
    unless trusted configuration explicitly requires one.
    Trusted transcript policy can require minimum trimmed character, UTF-8 byte
    and word counts or tighten the global 2000-character, 8000-byte and
    1000-word ceilings before any text reaches a classifier.
    Policy configuration is validated after authentication. Accepted text
    remains untrusted and is not logged.

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
        confidence=fields.get("Confidence"),
        minimum_confidence_percent=minimum_confidence_percent,
        require_confidence=require_confidence,
        min_transcript_characters=min_transcript_characters,
        max_transcript_characters=max_transcript_characters,
        min_transcript_utf8_bytes=min_transcript_utf8_bytes,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
        min_transcript_words=min_transcript_words,
        max_transcript_words=max_transcript_words,
    )


def resolve_claimed_speech_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str,
    workspace_id: str, step_id: str, claimer: CallbackClaimer,
    fallback_target: str, attempts: int, max_attempts: int = 3,
    minimum_confidence_percent: int | None = None,
    require_confidence: bool = False,
    min_transcript_characters: int = 1,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    min_transcript_utf8_bytes: int = 1,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
    min_transcript_words: int = 1,
    max_transcript_words: int = MAX_TRANSCRIPT_WORDS,
) -> SpeechDecision:
    """Validate trusted state, authenticate, claim, then admit speech once.

    Workspace, call, step, retry state and fallback must come from one trusted
    session snapshot. Policy is validated before authentication and claiming,
    so invalid configuration cannot consume a valid step. Rejected callbacks
    also leave it unclaimed, while authenticated replays fail before speech is
    inspected or returned. If configured, the trusted confidence presence and
    threshold rules are part of that policy snapshot; callback fields cannot
    replace them. Trusted transcript limits are snapshotted the same way.

    Production adapters still need an authorized current-step check and a
    durable conditional write, transactionally combined with related session
    state when required. This helper does not persist attempts, classify text,
    authorize model spend, render TwiML, acknowledge callbacks or perform I/O.
    """
    budget = _validate_speech_configuration(
        fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts,
        minimum_confidence_percent=minimum_confidence_percent,
        require_confidence=require_confidence,
        min_transcript_characters=min_transcript_characters,
        max_transcript_characters=max_transcript_characters,
        min_transcript_utf8_bytes=min_transcript_utf8_bytes,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
        min_transcript_words=min_transcript_words,
        max_transcript_words=max_transcript_words,
    )
    fields = validate_and_claim_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid, workspace_id=workspace_id,
        step_id=step_id, claimer=claimer,
    )
    return _resolve_validated_speech(
        fields.get("SpeechResult"), budget=budget, attempts=attempts,
        max_attempts=max_attempts, confidence=fields.get("Confidence"),
        minimum_confidence_percent=minimum_confidence_percent,
        require_confidence=require_confidence,
        min_transcript_characters=min_transcript_characters,
        max_transcript_characters=max_transcript_characters,
        min_transcript_utf8_bytes=min_transcript_utf8_bytes,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
        min_transcript_words=min_transcript_words,
        max_transcript_words=max_transcript_words,
    )
