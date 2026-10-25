"""Synthetic collection, forwarding, status, and hours demos without providers."""

import argparse
import json
from collections.abc import Callable, Iterable
from datetime import date, datetime
from typing import Literal

from zentomic.business_hours import resolve_business_hours_target_at
from zentomic.call_status import advance_call_status, is_terminal_call_status
from zentomic.classification import parse_intent_response
from zentomic.dial import resolve_dial_result
from zentomic.gather import GatherDecision, resolve_gather
from zentomic.intent import resolve_intent_confirmation
from zentomic.speech import (
    MAX_TRANSCRIPT_CHARS,
    MAX_TRANSCRIPT_LINES,
    MAX_TRANSCRIPT_UTF8_BYTES,
    MAX_TRANSCRIPT_WORDS,
    _validate_transcript_policy,
    resolve_speech_gather,
)
from zentomic.twiml import (
    SPEECH_RECOGNITION_LANGUAGES,
    SPEECH_RECOGNITION_MODELS,
    render_dtmf_gather,
    render_hangup,
    render_speech_gather,
)


_MAX_ATTEMPTS = 3


def _parse_speech_timeout(value: str | None) -> int | Literal["auto"]:
    """Parse the demo's bounded, non-reflecting speech timeout option."""
    if value is None:
        return 2
    if value == "auto":
        return "auto"
    if (value.isascii() and value.isdecimal() and not value.startswith("0")
            and len(value) <= 2):
        seconds = int(value)
        if seconds <= 60:
            return seconds
    raise ValueError("speech timeout must be auto or an integer from 1 to 60")


def _parse_speech_input_timeout(value: str | None) -> int:
    """Parse the demo's bounded wait for speech to begin."""
    if value is None:
        return 5
    if (value.isascii() and value.isdecimal() and not value.startswith("0")
            and len(value) <= 2):
        seconds = int(value)
        if seconds <= 60:
            return seconds
    raise ValueError("speech input timeout must be an integer from 1 to 60")


def _parse_speech_profanity_filter(value: str | None) -> bool | None:
    """Parse an explicit provider profanity-redaction preference."""
    if value is None:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("speech profanity filter must be true or false")


def _parse_speech_minimum_confidence(value: str | None) -> int | None:
    """Parse an optional canonical integer percentage without reflection."""
    if value is None:
        return None
    if value == "0":
        return 0
    if (value.isascii() and value.isdecimal() and not value.startswith("0")
            and len(value) <= 3):
        percent = int(value)
        if percent <= 100:
            return percent
    raise ValueError("minimum speech confidence must be an integer from 0 to 100")


def _parse_speech_max_characters(value: str | None) -> int:
    """Parse an optional stricter transcript ceiling without reflection."""
    if value is None:
        return MAX_TRANSCRIPT_CHARS
    if (len(value) <= 4 and value.isascii() and value.isdecimal()
            and not value.startswith("0")):
        characters = int(value)
        if characters <= MAX_TRANSCRIPT_CHARS:
            return characters
    raise ValueError("speech transcript limit must be an integer from 1 to 2000")


def _parse_speech_min_characters(value: str | None) -> int:
    """Parse an optional minimum trimmed transcript length privately."""
    if value is None:
        return 1
    if (len(value) <= 4 and value.isascii() and value.isdecimal()
            and not value.startswith("0")):
        characters = int(value)
        if characters <= MAX_TRANSCRIPT_CHARS:
            return characters
    raise ValueError("speech transcript minimum must be an integer from 1 to 2000")


def _parse_speech_max_utf8_bytes(value: str | None) -> int:
    """Parse an optional stricter UTF-8 transcript ceiling without reflection."""
    if value is None:
        return MAX_TRANSCRIPT_UTF8_BYTES
    if (len(value) <= 4 and value.isascii() and value.isdecimal()
            and not value.startswith("0")):
        byte_count = int(value)
        if byte_count <= MAX_TRANSCRIPT_UTF8_BYTES:
            return byte_count
    raise ValueError("speech transcript byte limit must be an integer from 1 to 8000")


def _parse_speech_min_utf8_bytes(value: str | None) -> int:
    """Parse an optional minimum trimmed UTF-8 transcript size privately."""
    if value is None:
        return 1
    if (len(value) <= 4 and value.isascii() and value.isdecimal()
            and not value.startswith("0")):
        byte_count = int(value)
        if byte_count <= MAX_TRANSCRIPT_UTF8_BYTES:
            return byte_count
    raise ValueError("speech transcript byte minimum must be an integer from 1 to 8000")


def _parse_speech_max_words(value: str | None) -> int:
    """Parse an optional whitespace-delimited transcript ceiling privately."""
    if value is None:
        return MAX_TRANSCRIPT_WORDS
    if (len(value) <= 4 and value.isascii() and value.isdecimal()
            and not value.startswith("0")):
        words = int(value)
        if words <= MAX_TRANSCRIPT_WORDS:
            return words
    raise ValueError("speech transcript word limit must be an integer from 1 to 1000")


def _parse_speech_min_words(value: str | None) -> int:
    """Parse an optional minimum whitespace-delimited word count privately."""
    if value is None:
        return 1
    if (len(value) <= 4 and value.isascii() and value.isdecimal()
            and not value.startswith("0")):
        words = int(value)
        if words <= MAX_TRANSCRIPT_WORDS:
            return words
    raise ValueError("speech transcript word minimum must be an integer from 1 to 1000")


def _parse_speech_max_lines(value: str | None) -> int:
    """Parse an optional Unicode-line transcript ceiling privately."""
    if value is None:
        return MAX_TRANSCRIPT_LINES
    if (len(value) <= 4 and value.isascii() and value.isdecimal()
            and not value.startswith("0")):
        lines = int(value)
        if lines <= MAX_TRANSCRIPT_LINES:
            return lines
    raise ValueError("speech transcript line limit must be an integer from 1 to 2000")


def _parse_speech_min_lines(value: str | None) -> int:
    """Parse an optional minimum Unicode-line transcript count privately."""
    if value is None:
        return 1
    if (len(value) <= 4 and value.isascii() and value.isdecimal()
            and not value.startswith("0")):
        lines = int(value)
        if lines <= MAX_TRANSCRIPT_LINES:
            return lines
    raise ValueError("speech transcript line minimum must be an integer from 1 to 2000")


def simulate_business_hours(
    instants: Iterable[datetime], *, timezone_name: str = "America/New_York",
) -> list[dict]:
    """Preview fixed synthetic weekday hours in an explicit workspace zone.

    Demo hours are Monday-Friday 09:00-17:00, with January 1, 2026 closed.
    The default zone is New York; the closure uses the selected zone's date.
    Each input is an independent timezone-aware clock fixture, not a callback
    or a live clock sample. Only synthetic target decisions are returned.
    No authentication, claims, state, TwiML, or dialing is performed.
    """
    return [{
        "action": "route",
        "target": resolve_business_hours_target_at(
            {day: [(540, 1020)] for day in range(5)},
            instant=instant, timezone_name=timezone_name,
            date_overrides={date(2026, 1, 1): []},
            open_target="demo-team", closed_target="demo-voicemail",
        ),
    } for instant in instants]


def simulate_call_status(
    statuses: Iterable[str], *, initial_status: str | None = None,
) -> list[dict]:
    """Replay synthetic observations for one leg using monotonic local state.

    Continue after terminal observations to illustrate delayed callbacks and
    validate every supplied status. A changed state is not permission to run
    cleanup. This does not authenticate, persist, deduplicate, or end a call.
    Optional initial_status simulates an already stored observation; validate
    it before consuming input, without emitting an extra observation step.
    Include validated incoming and previous statuses so an unchanged result
    can be distinguished as a duplicate, delayed, or conflicting observation.
    """
    if initial_status is not None:
        is_terminal_call_status(initial_status)
    current = initial_status
    steps = []
    for incoming in statuses:
        updated = advance_call_status(current, incoming)
        steps.append({
            "action": "observe-call-status", "status": updated,
            "previous_status": current, "incoming_status": incoming,
            "changed": updated != current,
            "terminal": is_terminal_call_status(updated),
        })
        current = updated
    return steps


def simulate_forwarding(statuses: Iterable[str]) -> list[dict]:
    """Apply at most two synthetic Number dial results to a fixed demo route.

    Start after target selection. Missing results leave the current attempt
    pending, never inventing a failure or success. Stop consuming on hangup;
    a failed fallback cannot recurse. Local fallback state only illustrates
    the policy, not atomic persistence, authenticated callbacks, or dialing.
    """
    steps = [{"action": "await-dial-result", "target": "demo-sales"}]
    inputs = iter(statuses)
    for fallback_used in (False, True):
        try:
            status = next(inputs)
        except StopIteration:
            break
        decision = resolve_dial_result(
            status, fallback_target="demo-reception", fallback_used=fallback_used,
        )
        if decision.action == "hangup":
            steps.append({"action": "hangup", "twiml": render_hangup()})
            break
        steps.append({"action": "fallback", "target": decision.target})
        steps.append({"action": "await-dial-result", "target": decision.target})
    return steps


def _simulate_collection(
    digits: Iterable[str], *, prompt: str, action_path: str,
    resolve: Callable[[str, int], GatherDecision],
) -> list[dict]:
    """Run either fixed demo policy with shared rendering and lazy consumption.

    The internal resolver must use the same three-attempt budget. Never read
    another input after routing, fallback, hangup, or budget exhaustion.
    """
    collection = render_dtmf_gather(prompt, action_path=action_path)
    steps = [{"action": "gather", "attempts": 0, "twiml": collection}]
    inputs = iter(digits)
    attempts = 0
    while attempts < _MAX_ATTEMPTS:
        decision = resolve(next(inputs, ""), attempts)
        attempts = decision.attempts
        step = {"action": decision.action, "attempts": attempts}
        if decision.target is not None:
            step["target"] = decision.target
        if decision.action == "retry":
            step["twiml"] = collection
        elif decision.action == "hangup":
            step["twiml"] = render_hangup()
        steps.append(step)
        if decision.action != "retry":
            break
    return steps


def simulate_keypad(digits: Iterable[str]) -> list[dict]:
    """Show a fixed demo menu, treating missing inputs as silence.

    Stop at routing, fallback, or hangup and consume at most three inputs.
    Return only synthetic targets and decisions, never the supplied digits.
    This is not an authenticated adapter or persistent call session. A route
    ends the simulation at target selection; it does not resolve or dial it.
    """
    return _simulate_collection(
        digits, prompt="Press 1 for sales, 2 for support, or 9 to end the call.",
        action_path="/voice/menu",
        resolve=lambda digit, attempts: resolve_gather(
            digit, {"1": "demo-sales", "2": "demo-support"},
            fallback_target="demo-reception", attempts=attempts,
            max_attempts=_MAX_ATTEMPTS, hangup_digit="9",
        ),
    )


def simulate_confirmation(intent: str, digits: Iterable[str]) -> list[dict]:
    """Confirm a fixed synthetic pending label, without classifying speech.

    Missing input is silence, never implicit confirmation. Stop at a selected
    opaque target or hangup, consuming at most three confirmation inputs.
    This demo's label is not authenticated session state or model output.
    """
    if not isinstance(intent, str) or intent not in ("sales", "support"):
        raise ValueError("intent must be sales or support")
    prompt = f"Did you mean {intent}? Press 1 for yes, 2 for reception, or 9 to end."
    return _simulate_collection(
        digits, prompt=prompt, action_path="/voice/confirm",
        resolve=lambda digit, attempts: resolve_intent_confirmation(
            digit, intent, {"sales": "demo-sales", "support": "demo-support"},
            fallback_target="demo-reception", attempts=attempts,
            max_attempts=_MAX_ATTEMPTS, hangup_digit="9",
        ),
    )


def simulate_classification(response: str | None, digits: Iterable[str]) -> list[dict]:
    """Admit synthetic model output, then require separate keypad confirmation.

    Unusable output falls back without collecting input or reflecting the
    response. This only parses a supplied fixture; it never invokes a model.
    """
    intent = parse_intent_response(response, allowed_intents=("sales", "support"))
    if intent is None:
        return [{"action": "fallback", "attempts": 0, "target": "demo-reception"}]
    return simulate_confirmation(intent, digits)


def simulate_speech(
    speeches: Iterable[str | None], response: str | None, digits: Iterable[str],
    *, hints: list[str] | tuple[str, ...] | None = None, language: str = "en-US",
    input_timeout: int = 5, speech_timeout: int | Literal["auto"] = 2,
    profanity_filter: bool | None = None, speech_model: str | None = None,
    confidences: Iterable[str | None] | None = None,
    minimum_confidence_percent: int | None = None,
    require_confidence: bool = False,
    min_transcript_characters: int = 1,
    max_transcript_characters: int = MAX_TRANSCRIPT_CHARS,
    min_transcript_utf8_bytes: int = 1,
    max_transcript_utf8_bytes: int = MAX_TRANSCRIPT_UTF8_BYTES,
    min_transcript_words: int = 1,
    max_transcript_words: int = MAX_TRANSCRIPT_WORDS,
    min_transcript_lines: int = 1,
    max_transcript_lines: int = MAX_TRANSCRIPT_LINES,
    require_transcript_alphanumeric: bool = False,
) -> list[dict]:
    """Admit synthetic speech, then a supplied classifier fixture and keypad input.

    Each collection stage has its own three-attempt budget. Stop consuming
    speech on admission or exhaustion; never parse the response or consume
    digits if speech was not admitted. Transcript and raw response stay out
    of the returned steps. This does not transcribe or run a classifier, and
    the fixture need not represent the supplied speech's actual meaning.
    Optional synthetic vocabulary is validated before consuming any inputs,
    appears in collection XML, and never changes the classifier fixture.
    The explicit recognition language changes only the speech Gather XML; it
    does not translate prompts, classifier fixtures, or confirmation prompts.
    The input timeout controls how long collection waits for speech to begin.
    The speech timeout controls the simulated collection XML only; ``auto``
    asks a future provider to stop at the first pause, not after a fixed limit.
    An explicit profanity filter changes recognition output policy only; it
    does not inspect, rewrite, or return the supplied synthetic transcript.
    An explicit speech model changes only the collection XML and does not run
    recognition or select a provider.
    Optional confidence fixtures align with speech attempts. When a trusted
    minimum is configured, explicit scores below it retry. Missing scores stay
    admissible unless trusted configuration requires one. Neither scores nor
    transcripts are returned.
    Trusted transcript policy can require minimum trimmed character, UTF-8 byte,
    word, and line counts or reduce the fixed 2000-character, 8000-byte,
    1000-word and 2000-line admission bounds without changing collection XML or returning
    rejected text. Words are separated by Unicode whitespace; lines use
    Python's Unicode line-boundary rules. Optional alphanumeric policy retries
    punctuation-, symbol-, and mark-only transcripts before classification.
    """
    if (minimum_confidence_percent is not None
            and (type(minimum_confidence_percent) is not int
                 or not 0 <= minimum_confidence_percent <= 100)):
        raise ValueError("minimum_confidence_percent must be an integer from 0 to 100 or None")
    if type(require_confidence) is not bool:
        raise ValueError("require_confidence must be a boolean")
    _validate_transcript_policy(
        min_transcript_characters=min_transcript_characters,
        max_transcript_characters=max_transcript_characters,
        min_transcript_utf8_bytes=min_transcript_utf8_bytes,
        max_transcript_utf8_bytes=max_transcript_utf8_bytes,
        min_transcript_words=min_transcript_words,
        max_transcript_words=max_transcript_words,
        min_transcript_lines=min_transcript_lines,
        max_transcript_lines=max_transcript_lines,
        require_transcript_alphanumeric=require_transcript_alphanumeric,
    )
    collection = render_speech_gather(
        "How can we help? Say sales or support.", action_path="/voice/speech",
        timeout=input_timeout, hints=hints, language=language,
        speech_timeout=speech_timeout, profanity_filter=profanity_filter,
        speech_model=speech_model,
    )
    steps = [{"stage": "speech", "action": "gather", "attempts": 0, "twiml": collection}]
    inputs = iter(speeches)
    confidence_inputs = iter(()) if confidences is None else iter(confidences)
    attempts = 0
    while attempts < _MAX_ATTEMPTS:
        decision = resolve_speech_gather(
            next(inputs, None), fallback_target="demo-reception",
            attempts=attempts, max_attempts=_MAX_ATTEMPTS,
            confidence=next(confidence_inputs, None),
            minimum_confidence_percent=minimum_confidence_percent,
            require_confidence=require_confidence,
            min_transcript_characters=min_transcript_characters,
            max_transcript_characters=max_transcript_characters,
            min_transcript_utf8_bytes=min_transcript_utf8_bytes,
            max_transcript_utf8_bytes=max_transcript_utf8_bytes,
            min_transcript_words=min_transcript_words,
            max_transcript_words=max_transcript_words,
            min_transcript_lines=min_transcript_lines,
            max_transcript_lines=max_transcript_lines,
            require_transcript_alphanumeric=require_transcript_alphanumeric,
        )
        attempts = decision.attempts
        step = {"stage": "speech", "action": decision.action, "attempts": attempts}
        if decision.target is not None:
            step["target"] = decision.target
        if decision.action == "retry":
            step["twiml"] = collection
        steps.append(step)
        if decision.action == "classify":
            steps.extend(dict(item, stage="confirmation")
                         for item in simulate_classification(response, digits))
            break
        if decision.action == "fallback":
            break
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "digits", nargs="*",
        help="synthetic inputs; menu: 1 sales, 2 support, 9 hangup; use '' for silence",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--intent", choices=("sales", "support"),
        help="confirm a synthetic pending intent: 1 yes, 2 reception, 9 hangup",
    )
    mode.add_argument(
        "--classifier-response", metavar="JSON",
        help="admit synthetic classifier JSON before confirmation; never calls a model",
    )
    mode.add_argument(
        "--dial-result", action="append", metavar="STATUS",
        help="synthetic Number dial result; repeat for one fallback; never dials",
    )
    mode.add_argument(
        "--call-status", action="append", metavar="STATUS",
        help="replay synthetic single-leg lifecycle observations; never changes a real call",
    )
    mode.add_argument(
        "--business-hours-at", action="append", metavar="ISO_DATETIME",
        help="preview fixed weekday hours at an aware ISO datetime; repeat for samples",
    )
    parser.add_argument(
        "--business-hours-timezone", metavar="IANA_ZONE",
        help="workspace zone for --business-hours-at (default: America/New_York)",
    )
    parser.add_argument(
        "--initial-call-status", metavar="STATUS",
        help="synthetic stored lifecycle state; requires --call-status; never loads a session",
    )
    parser.add_argument(
        "--speech", action="append", metavar="TEXT",
        help="synthetic transcript in order; repeat for retries; requires --classifier-response",
    )
    parser.add_argument(
        "--speech-hint", action="append", metavar="PHRASE",
        help="synthetic recognition vocabulary in collection XML; repeat; requires --speech",
    )
    parser.add_argument(
        "--speech-language", metavar="LOCALE",
        help="recognition locale for speech collection XML (default: en-US); requires --speech",
    )
    parser.add_argument(
        "--speech-timeout", metavar="SECONDS|auto",
        help="pause ending speech collection (1-60 or auto; default: 2); requires --speech",
    )
    parser.add_argument(
        "--speech-input-timeout", metavar="SECONDS",
        help="wait for speech to begin (1-60; default: 5); requires --speech",
    )
    parser.add_argument(
        "--speech-profanity-filter", metavar="true|false",
        help="explicit recognition redaction policy; requires --speech",
    )
    parser.add_argument(
        "--speech-model", metavar="MODEL",
        help="generic recognition model for collection XML; requires --speech",
    )
    parser.add_argument(
        "--speech-confidence", action="append", metavar="SCORE",
        help="synthetic provider score aligned with each --speech; repeat",
    )
    parser.add_argument(
        "--speech-minimum-confidence", metavar="PERCENT",
        help="trusted admission threshold from 0 to 100; requires --speech",
    )
    parser.add_argument(
        "--speech-require-confidence", action="store_true",
        help="retry when the synthetic provider score is missing; requires --speech",
    )
    parser.add_argument(
        "--speech-require-alphanumeric", action="store_true",
        help="retry transcripts without a Unicode letter or number; requires --speech",
    )
    parser.add_argument(
        "--speech-min-characters", metavar="COUNT",
        help="trusted trimmed transcript minimum from 1 to 2000; requires --speech",
    )
    parser.add_argument(
        "--speech-max-characters", metavar="COUNT",
        help="trusted transcript ceiling from 1 to 2000; requires --speech",
    )
    parser.add_argument(
        "--speech-min-bytes", metavar="COUNT",
        help="trusted trimmed UTF-8 minimum from 1 to 8000; requires --speech",
    )
    parser.add_argument(
        "--speech-max-bytes", metavar="COUNT",
        help="trusted UTF-8 transcript ceiling from 1 to 8000; requires --speech",
    )
    parser.add_argument(
        "--speech-min-words", metavar="COUNT",
        help="trusted whitespace-delimited minimum from 1 to 1000; requires --speech",
    )
    parser.add_argument(
        "--speech-max-words", metavar="COUNT",
        help="trusted whitespace-delimited ceiling from 1 to 1000; requires --speech",
    )
    parser.add_argument(
        "--speech-min-lines", metavar="COUNT",
        help="trusted Unicode-line minimum from 1 to 2000; requires --speech",
    )
    parser.add_argument(
        "--speech-max-lines", metavar="COUNT",
        help="trusted Unicode-line ceiling from 1 to 2000; requires --speech",
    )
    args = parser.parse_args(argv)
    if args.speech_hint is not None and args.speech is None:
        parser.error("--speech-hint requires --speech")
    if args.speech_language is not None and args.speech is None:
        parser.error("--speech-language requires --speech")
    if args.speech_timeout is not None and args.speech is None:
        parser.error("--speech-timeout requires --speech")
    if args.speech_input_timeout is not None and args.speech is None:
        parser.error("--speech-input-timeout requires --speech")
    if args.speech_profanity_filter is not None and args.speech is None:
        parser.error("--speech-profanity-filter requires --speech")
    if args.speech_model is not None and args.speech is None:
        parser.error("--speech-model requires --speech")
    if args.speech_confidence is not None and args.speech is None:
        parser.error("--speech-confidence requires --speech")
    if args.speech_minimum_confidence is not None and args.speech is None:
        parser.error("--speech-minimum-confidence requires --speech")
    if args.speech_require_confidence and args.speech is None:
        parser.error("--speech-require-confidence requires --speech")
    if args.speech_require_alphanumeric and args.speech is None:
        parser.error("--speech-require-alphanumeric requires --speech")
    if args.speech_min_characters is not None and args.speech is None:
        parser.error("--speech-min-characters requires --speech")
    if args.speech_max_characters is not None and args.speech is None:
        parser.error("--speech-max-characters requires --speech")
    if args.speech_min_bytes is not None and args.speech is None:
        parser.error("--speech-min-bytes requires --speech")
    if args.speech_max_bytes is not None and args.speech is None:
        parser.error("--speech-max-bytes requires --speech")
    if args.speech_min_words is not None and args.speech is None:
        parser.error("--speech-min-words requires --speech")
    if args.speech_max_words is not None and args.speech is None:
        parser.error("--speech-max-words requires --speech")
    if args.speech_min_lines is not None and args.speech is None:
        parser.error("--speech-min-lines requires --speech")
    if args.speech_max_lines is not None and args.speech is None:
        parser.error("--speech-max-lines requires --speech")
    if (args.speech_confidence is not None
            and args.speech_minimum_confidence is None
            and not args.speech_require_confidence):
        parser.error("--speech-confidence requires --speech-minimum-confidence or "
                     "--speech-require-confidence")
    if (args.speech_confidence is not None
            and len(args.speech_confidence) != len(args.speech)):
        parser.error("provide one --speech-confidence for each --speech")
    if (args.speech_language is not None
            and args.speech_language not in SPEECH_RECOGNITION_LANGUAGES):
        parser.error("invalid synthetic speech language; use a supported project locale")
    if (args.speech_model is not None
            and args.speech_model not in SPEECH_RECOGNITION_MODELS):
        parser.error("invalid synthetic speech model; use a supported generic model")
    try:
        speech_timeout = _parse_speech_timeout(args.speech_timeout)
    except ValueError:
        parser.error("invalid synthetic speech timeout; use auto or an integer from 1 to 60")
    try:
        speech_input_timeout = _parse_speech_input_timeout(args.speech_input_timeout)
    except ValueError:
        parser.error("invalid synthetic speech input timeout; use an integer from 1 to 60")
    try:
        speech_profanity_filter = _parse_speech_profanity_filter(
            args.speech_profanity_filter,
        )
    except ValueError:
        parser.error("invalid synthetic speech profanity filter; use true or false")
    try:
        speech_minimum_confidence = _parse_speech_minimum_confidence(
            args.speech_minimum_confidence,
        )
    except ValueError:
        parser.error("invalid synthetic speech minimum confidence; use an integer from 0 to 100")
    try:
        speech_min_characters = _parse_speech_min_characters(
            args.speech_min_characters,
        )
    except ValueError:
        parser.error("invalid synthetic speech transcript minimum; use an integer from 1 to 2000")
    try:
        speech_max_characters = _parse_speech_max_characters(
            args.speech_max_characters,
        )
    except ValueError:
        parser.error("invalid synthetic speech transcript limit; use an integer from 1 to 2000")
    try:
        speech_min_utf8_bytes = _parse_speech_min_utf8_bytes(
            args.speech_min_bytes,
        )
    except ValueError:
        parser.error("invalid synthetic speech transcript byte minimum; use an integer from 1 to 8000")
    try:
        speech_max_utf8_bytes = _parse_speech_max_utf8_bytes(
            args.speech_max_bytes,
        )
    except ValueError:
        parser.error("invalid synthetic speech transcript byte limit; use an integer from 1 to 8000")
    try:
        speech_min_words = _parse_speech_min_words(args.speech_min_words)
    except ValueError:
        parser.error("invalid synthetic speech transcript word minimum; use an integer from 1 to 1000")
    try:
        speech_max_words = _parse_speech_max_words(args.speech_max_words)
    except ValueError:
        parser.error("invalid synthetic speech transcript word limit; use an integer from 1 to 1000")
    try:
        speech_min_lines = _parse_speech_min_lines(args.speech_min_lines)
    except ValueError:
        parser.error("invalid synthetic speech transcript line minimum; use an integer from 1 to 2000")
    try:
        speech_max_lines = _parse_speech_max_lines(args.speech_max_lines)
    except ValueError:
        parser.error("invalid synthetic speech transcript line limit; use an integer from 1 to 2000")
    if speech_min_characters > speech_max_characters:
        parser.error("synthetic speech transcript minimum cannot exceed its maximum")
    if speech_min_words > speech_max_words:
        parser.error("synthetic speech transcript word minimum cannot exceed its maximum")
    if speech_min_utf8_bytes > speech_max_utf8_bytes:
        parser.error("synthetic speech transcript byte minimum cannot exceed its maximum")
    if speech_min_lines > speech_max_lines:
        parser.error("synthetic speech transcript line minimum cannot exceed its maximum")
    if speech_min_characters > speech_max_utf8_bytes:
        parser.error(
            "synthetic speech transcript minimum cannot fit within its byte limit"
        )
    minimum_word_characters = speech_min_words * 2 - 1
    if (minimum_word_characters > speech_max_characters
            or minimum_word_characters > speech_max_utf8_bytes):
        parser.error(
            "synthetic speech transcript word minimum cannot fit within its "
            "character and byte limits"
        )
    if (speech_min_lines > speech_max_characters
            or speech_min_lines > speech_max_utf8_bytes):
        parser.error(
            "synthetic speech transcript line minimum cannot fit within its "
            "character and byte limits"
        )
    minimum_separators = max(speech_min_words - 1, speech_min_lines - 1)
    minimum_structural_characters = speech_min_words + minimum_separators
    if speech_min_words == 1 and speech_min_lines > 1:
        leading_line_characters = speech_min_characters + speech_min_lines - 1
        minimum_policy_characters = leading_line_characters
        if speech_max_words >= 2:
            minimum_policy_characters = min(
                leading_line_characters,
                max(speech_min_characters, speech_min_lines + 1),
            )
    else:
        minimum_policy_characters = max(
            speech_min_characters, minimum_structural_characters,
        )
    if (minimum_policy_characters > speech_max_characters
            or minimum_policy_characters > speech_max_utf8_bytes):
        parser.error(
            "synthetic speech transcript word and line minimums cannot fit within "
            "their character and byte limits"
        )
    if speech_max_words == 1 and speech_min_lines > 1:
        maximum_content_bytes = (
            speech_max_characters - (speech_min_lines - 1)
        ) * 4
    else:
        maximum_content_bytes = (
            speech_max_characters * 4 - minimum_separators
        )
    if speech_min_utf8_bytes > maximum_content_bytes:
        parser.error(
            "synthetic speech transcript byte minimum cannot fit within its "
            "character and word limits"
        )
    if args.business_hours_timezone is not None and args.business_hours_at is None:
        parser.error("--business-hours-timezone requires --business-hours-at")
    if args.business_hours_at is not None and (args.digits or args.speech is not None):
        parser.error("--business-hours-at cannot be combined with keypad or speech inputs")
    if args.initial_call_status is not None and args.call_status is None:
        parser.error("--initial-call-status requires --call-status")
    if args.call_status is not None and (args.digits or args.speech is not None):
        parser.error("--call-status cannot be combined with keypad or speech inputs")
    if args.dial_result is not None and (args.digits or args.speech is not None):
        parser.error("--dial-result cannot be combined with keypad or speech inputs")
    if args.speech is not None and args.classifier_response is None:
        parser.error("--speech requires --classifier-response")
    # No arguments demonstrate a silent attempt followed by a valid selection.
    if args.business_hours_at is not None:
        try:
            steps = simulate_business_hours(
                (datetime.fromisoformat(value) for value in args.business_hours_at),
                timezone_name=("America/New_York" if args.business_hours_timezone is None
                               else args.business_hours_timezone),
            )
        except (ValueError, OverflowError):
            parser.error("invalid synthetic business-hours instant or unavailable demo timezone")
    elif args.call_status is not None:
        try:
            steps = simulate_call_status(
                args.call_status, initial_status=args.initial_call_status,
            )
        except ValueError:
            parser.error("unsupported synthetic call status")
    elif args.dial_result is not None:
        try:
            steps = simulate_forwarding(args.dial_result)
        except ValueError:
            parser.error("unsupported synthetic Number dial result status")
    elif args.speech is not None:
        try:
            steps = simulate_speech(
                args.speech, args.classifier_response, args.digits, hints=args.speech_hint,
                language=("en-US" if args.speech_language is None else args.speech_language),
                input_timeout=speech_input_timeout, speech_timeout=speech_timeout,
                profanity_filter=speech_profanity_filter, speech_model=args.speech_model,
                confidences=args.speech_confidence,
                minimum_confidence_percent=speech_minimum_confidence,
                require_confidence=args.speech_require_confidence,
                min_transcript_characters=speech_min_characters,
                max_transcript_characters=speech_max_characters,
                min_transcript_utf8_bytes=speech_min_utf8_bytes,
                max_transcript_utf8_bytes=speech_max_utf8_bytes,
                min_transcript_words=speech_min_words,
                max_transcript_words=speech_max_words,
                min_transcript_lines=speech_min_lines,
                max_transcript_lines=speech_max_lines,
                require_transcript_alphanumeric=args.speech_require_alphanumeric,
            )
        except ValueError:
            parser.error("invalid synthetic speech hints: use at most 500 nonblank single-line "
                         "XML phrases of at most 100 characters each, without commas")
    elif args.classifier_response is not None:
        steps = simulate_classification(args.classifier_response, args.digits)
    elif args.intent is not None:
        steps = simulate_confirmation(args.intent, args.digits)
    else:
        steps = simulate_keypad(args.digits or ["", "1"])
    print(json.dumps({"mode": "offline-demo", "steps": steps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
