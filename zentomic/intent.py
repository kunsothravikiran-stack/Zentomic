"""Pure routing of confirmed intent labels, without AI or telephony calls."""

from collections.abc import Mapping
from itertools import islice

from zentomic.gather import GatherDecision, _resolve_validated_gather
from zentomic.labels import _valid_intent_label
from zentomic.routing import _valid_target


MAX_INTENT_ROUTES = 128


def _snapshot_intent_routes(routes: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(routes, Mapping):
        raise ValueError("routes must be a mapping")
    # Read one extra key before copying values so malformed workspace
    # configuration cannot cause an unbounded snapshot or validation pass.
    keys = tuple(islice(routes, MAX_INTENT_ROUTES + 1))
    if len(keys) > MAX_INTENT_ROUTES:
        raise ValueError("routes must contain at most 128 entries")
    return {key: routes[key] for key in keys}


def _validate_intent_routes(
    routes: Mapping[str, str], *, fallback_target: str,
) -> dict[str, str]:
    """Return a private route snapshot after validating trusted destinations."""
    if not _valid_target(fallback_target):
        raise ValueError("fallback_target must be a nonblank UTF-8 string of at most 256 bytes")

    menu = _snapshot_intent_routes(routes)
    for label, target in menu.items():
        if not _valid_intent_label(label):
            raise ValueError(
                "intent labels must be nonblank UTF-8 strings of at most 256 bytes"
            )
        if not _valid_target(target):
            raise ValueError("route targets must be nonblank UTF-8 strings of at most 256 bytes")
    return menu


def resolve_intent(
    intent: str | None,
    routes: Mapping[str, str],
    *,
    fallback_target: str,
    confirmed: bool = False,
) -> str:
    """Select an allowlisted target only after explicit caller confirmation.

    Intent labels must be nonblank UTF-8 strings of at most 256 bytes and match
    exactly, without trimming, case folding, or coercion, consistent with
    classifier allowlist validation.
    Missing, malformed, unknown, or unconfirmed intents use the fallback.
    Only the boolean True counts as confirmation, not truthy strings/numbers.
    Invalid configuration raises ValueError before any route is selected.

    This helper does not classify speech or establish caller confirmation.
    Integrators must obtain confirmation from trusted call-session state and
    authorize every target, including fallback, within the current workspace.
    """
    menu = _validate_intent_routes(routes, fallback_target=fallback_target)

    if confirmed is not True or not _valid_intent_label(intent):
        return fallback_target
    return menu.get(intent, fallback_target)


def _validate_intent_confirmation_configuration(
    intent: str | None, routes: Mapping[str, str], *, fallback_target: str,
    attempts: int, max_attempts: int, hangup_digit: str | None,
) -> tuple[dict[str, str], bool]:
    """Snapshot and validate trusted confirmation state without consuming input."""
    if type(attempts) is not int or attempts < 0:
        raise ValueError("attempts must be a nonnegative integer")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    menu = _validate_intent_routes(routes, fallback_target=fallback_target)
    if hangup_digit is not None:
        if (not isinstance(hangup_digit, str) or len(hangup_digit) != 1
                or hangup_digit not in "0123456789"):
            raise ValueError("hangup_digit must be a single ASCII digit or None")
        if hangup_digit in ("1", "2"):
            raise ValueError("hangup_digit must not overlap a configured route")

    known = _valid_intent_label(intent) and intent in menu
    target = menu[intent] if known else fallback_target
    return {"1": target, "2": fallback_target}, known


def _resolve_validated_intent_confirmation(
    digits: str | None, confirmation_menu: Mapping[str, str], *, known: bool,
    fallback_target: str, attempts: int, max_attempts: int,
    hangup_digit: str | None,
) -> GatherDecision:
    """Resolve caller input after trusted confirmation state was validated."""
    decision = _resolve_validated_gather(
        digits, confirmation_menu, fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
    if not known or digits == "2":
        return GatherDecision("fallback", fallback_target, decision.attempts)
    return decision


def resolve_intent_confirmation(
    digits: str | None,
    intent: str | None,
    routes: Mapping[str, str],
    *,
    fallback_target: str,
    attempts: int,
    max_attempts: int = 3,
    hangup_digit: str | None = None,
) -> GatherDecision:
    """Resolve a bounded keypad confirmation for a trusted pending intent.

    1 confirms the proposed route; 2 declines it and selects a human fallback.
    Silence or invalid input retries within the existing collection budget.
    Missing/unknown intents and exhausted budgets always fall back. Every
    completed collection consumes an attempt unless already exhausted.
    An optional ASCII digit other than 1 or 2 ends an active, known-intent
    confirmation with no destination. It follows the same collection budget.

    The pending intent and attempt count must come from authenticated,
    workspace-scoped state for this exact confirmation step, not the webhook
    or classifier's claim of confirmation. This does not persist or deduplicate.
    """
    confirmation_menu, known = _validate_intent_confirmation_configuration(
        intent, routes, fallback_target=fallback_target, attempts=attempts,
        max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
    return _resolve_validated_intent_confirmation(
        digits, confirmation_menu, known=known, fallback_target=fallback_target,
        attempts=attempts, max_attempts=max_attempts, hangup_digit=hangup_digit,
    )
