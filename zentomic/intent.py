"""Pure routing of confirmed intent labels, without AI or telephony calls."""

from collections.abc import Mapping

from zentomic.gather import GatherDecision, resolve_gather


def resolve_intent(
    intent: str | None,
    routes: Mapping[str, str],
    *,
    fallback_target: str,
    confirmed: bool = False,
) -> str:
    """Select an allowlisted target only after explicit caller confirmation.

    Intent labels match exactly, without trimming, case folding, or coercion.
    Missing, malformed, unknown, or unconfirmed intents use the fallback.
    Only the boolean True counts as confirmation, not truthy strings/numbers.
    Invalid configuration raises ValueError before any route is selected.

    This helper does not classify speech or establish caller confirmation.
    Integrators must obtain confirmation from trusted call-session state and
    authorize every target, including fallback, within the current workspace.
    """
    if not isinstance(routes, Mapping):
        raise ValueError("routes must be a mapping")
    if not isinstance(fallback_target, str) or not fallback_target.strip():
        raise ValueError("fallback_target must be a nonblank string")

    menu = dict(routes)
    for label, target in menu.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("intent labels must be nonblank strings")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("route targets must be nonblank strings")

    if confirmed is not True or not isinstance(intent, str):
        return fallback_target
    return menu.get(intent, fallback_target)


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
    if not isinstance(routes, Mapping):
        raise ValueError("routes must be a mapping")
    menu = dict(routes)
    target = resolve_intent(intent, menu, fallback_target=fallback_target, confirmed=True)
    known = isinstance(intent, str) and intent in menu
    decision = resolve_gather(
        digits, {"1": target, "2": fallback_target},
        fallback_target=fallback_target, attempts=attempts, max_attempts=max_attempts,
        hangup_digit=hangup_digit,
    )
    if not known or digits == "2":
        return GatherDecision("fallback", fallback_target, decision.attempts)
    return decision
