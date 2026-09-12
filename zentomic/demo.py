"""Synthetic collection, forwarding, and call-status demos without providers."""

import argparse
import json
from collections.abc import Callable, Iterable

from zentomic.call_status import advance_call_status, is_terminal_call_status
from zentomic.classification import parse_intent_response
from zentomic.dial import resolve_dial_result
from zentomic.gather import GatherDecision, resolve_gather
from zentomic.intent import resolve_intent_confirmation
from zentomic.speech import resolve_speech_gather
from zentomic.twiml import render_dtmf_gather, render_hangup, render_speech_gather


_MAX_ATTEMPTS = 3


def simulate_call_status(
    statuses: Iterable[str], *, initial_status: str | None = None,
) -> list[dict]:
    """Replay synthetic observations for one leg using monotonic local state.

    Continue after terminal observations to illustrate delayed callbacks and
    validate every supplied status. A changed state is not permission to run
    cleanup. This does not authenticate, persist, deduplicate, or end a call.
    Optional initial_status simulates an already stored observation; validate
    it before consuming input, without emitting an extra observation step.
    """
    if initial_status is not None:
        is_terminal_call_status(initial_status)
    current = initial_status
    steps = []
    for incoming in statuses:
        updated = advance_call_status(current, incoming)
        steps.append({
            "action": "observe-call-status", "status": updated,
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
) -> list[dict]:
    """Admit synthetic speech, then a supplied classifier fixture and keypad input.

    Each collection stage has its own three-attempt budget. Stop consuming
    speech on admission or exhaustion; never parse the response or consume
    digits if speech was not admitted. Transcript and raw response stay out
    of the returned steps. This does not transcribe or run a classifier, and
    the fixture need not represent the supplied speech's actual meaning.
    """
    collection = render_speech_gather(
        "How can we help? Say sales or support.", action_path="/voice/speech",
    )
    steps = [{"stage": "speech", "action": "gather", "attempts": 0, "twiml": collection}]
    inputs = iter(speeches)
    attempts = 0
    while attempts < _MAX_ATTEMPTS:
        decision = resolve_speech_gather(
            next(inputs, None), fallback_target="demo-reception",
            attempts=attempts, max_attempts=_MAX_ATTEMPTS,
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
    parser.add_argument(
        "--initial-call-status", metavar="STATUS",
        help="synthetic stored lifecycle state; requires --call-status; never loads a session",
    )
    parser.add_argument(
        "--speech", action="append", metavar="TEXT",
        help="synthetic transcript in order; repeat for retries; requires --classifier-response",
    )
    args = parser.parse_args(argv)
    if args.initial_call_status is not None and args.call_status is None:
        parser.error("--initial-call-status requires --call-status")
    if args.call_status is not None and (args.digits or args.speech is not None):
        parser.error("--call-status cannot be combined with keypad or speech inputs")
    if args.dial_result is not None and (args.digits or args.speech is not None):
        parser.error("--dial-result cannot be combined with keypad or speech inputs")
    if args.speech is not None and args.classifier_response is None:
        parser.error("--speech requires --classifier-response")
    # No arguments demonstrate a silent attempt followed by a valid selection.
    if args.call_status is not None:
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
        steps = simulate_speech(args.speech, args.classifier_response, args.digits)
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
