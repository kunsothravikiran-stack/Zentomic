"""Pure routing of confirmed intent labels, without AI or telephony calls."""

from collections.abc import Mapping


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
