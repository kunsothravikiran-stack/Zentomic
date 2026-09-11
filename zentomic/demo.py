"""Synthetic keypad and intent-confirmation demos with no provider calls."""

import argparse
import json
from collections.abc import Iterable

from zentomic.gather import resolve_gather
from zentomic.intent import resolve_intent_confirmation
from zentomic.twiml import render_dtmf_gather, render_hangup


def simulate_keypad(digits: Iterable[str]) -> list[dict]:
    """Show a fixed demo menu, treating missing inputs as silence.

    Stop at routing, fallback, or hangup and consume at most three inputs.
    Return only synthetic targets and decisions, never the supplied digits.
    This is not an authenticated adapter or persistent call session. A route
    ends the simulation at target selection; it does not resolve or dial it.
    """
    prompt = "Press 1 for sales, 2 for support, or 9 to end the call."
    collection = render_dtmf_gather(prompt, action_path="/voice/menu")
    steps = [{"action": "gather", "attempts": 0, "twiml": collection}]
    inputs = iter(digits)
    attempts = 0
    while attempts < 3:
        decision = resolve_gather(
            next(inputs, ""), {"1": "demo-sales", "2": "demo-support"},
            fallback_target="demo-reception", attempts=attempts,
            max_attempts=3, hangup_digit="9",
        )
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


def simulate_confirmation(intent: str, digits: Iterable[str]) -> list[dict]:
    """Confirm a fixed synthetic pending label, without classifying speech.

    Missing input is silence, never implicit confirmation. Stop at a selected
    opaque target or hangup, consuming at most three confirmation inputs.
    This demo's label is not authenticated session state or model output.
    """
    if not isinstance(intent, str) or intent not in ("sales", "support"):
        raise ValueError("intent must be sales or support")
    prompt = f"Did you mean {intent}? Press 1 for yes, 2 for reception, or 9 to end."
    collection = render_dtmf_gather(prompt, action_path="/voice/confirm")
    steps = [{"action": "gather", "attempts": 0, "twiml": collection}]
    inputs = iter(digits)
    attempts = 0
    while attempts < 3:
        decision = resolve_intent_confirmation(
            next(inputs, ""), intent, {"sales": "demo-sales", "support": "demo-support"},
            fallback_target="demo-reception", attempts=attempts,
            max_attempts=3, hangup_digit="9",
        )
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "digits", nargs="*",
        help="synthetic inputs; menu: 1 sales, 2 support, 9 hangup; use '' for silence",
    )
    parser.add_argument(
        "--intent", choices=("sales", "support"),
        help="confirm a synthetic pending intent: 1 yes, 2 reception, 9 hangup",
    )
    args = parser.parse_args(argv)
    # No arguments demonstrate a silent attempt followed by a valid selection.
    steps = (simulate_confirmation(args.intent, args.digits) if args.intent is not None
             else simulate_keypad(args.digits or ["", "1"]))
    print(json.dumps({"mode": "offline-demo", "steps": steps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
