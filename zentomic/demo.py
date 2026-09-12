"""Synthetic keypad and intent-confirmation demos with no provider calls."""

import argparse
import json
from collections.abc import Callable, Iterable

from zentomic.classification import parse_intent_response
from zentomic.gather import GatherDecision, resolve_gather
from zentomic.intent import resolve_intent_confirmation
from zentomic.twiml import render_dtmf_gather, render_hangup


_MAX_ATTEMPTS = 3


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
    args = parser.parse_args(argv)
    # No arguments demonstrate a silent attempt followed by a valid selection.
    if args.classifier_response is not None:
        steps = simulate_classification(args.classifier_response, args.digits)
    elif args.intent is not None:
        steps = simulate_confirmation(args.intent, args.digits)
    else:
        steps = simulate_keypad(args.digits or ["", "1"])
    print(json.dumps({"mode": "offline-demo", "steps": steps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
