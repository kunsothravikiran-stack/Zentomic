"""Authenticated process-local callback claims for offline adapter tests."""

from collections.abc import Mapping
from threading import Lock
from typing import Any, Callable, Protocol

from zentomic.authentication import SignatureValidator, validate_call_event


MAX_CLAIM_IDENTIFIER_BYTES = 256
_IDENTIFIER_ERROR = (
    "claim identifiers must be nonblank UTF-8 strings of at most 256 bytes"
)


class CallbackClaimCapacityError(ValueError):
    """A new callback cannot be claimed without exceeding this instance's limit."""


class CallbackReplayError(ValueError):
    """An authenticated callback repeats an already claimed synthetic step."""


class CallbackClaimer(Protocol):
    """Minimal atomic claim boundary supplied by a local or durable adapter."""

    def claim(self, workspace_id: str, call_sid: str, step_id: str) -> bool:
        """Return exactly True for a new key and exactly False for a replay."""


PreparedCallbackClaim = tuple[
    Callable[[str, str, str], bool],
    tuple[str, str, str],
]


def _key(workspace_id: str, call_sid: str, step_id: str) -> tuple[str, str, str]:
    for value in (workspace_id, call_sid, step_id):
        # UTF-8 needs at least one byte per character. Reject long inputs before
        # stripping or encoding to avoid work proportional to an arbitrary input.
        if (not isinstance(value, str) or len(value) > MAX_CLAIM_IDENTIFIER_BYTES
                or not value.strip()):
            raise ValueError(_IDENTIFIER_ERROR)
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(_IDENTIFIER_ERROR) from None
        if len(encoded) > MAX_CLAIM_IDENTIFIER_BYTES:
            raise ValueError(_IDENTIFIER_ERROR)
    return workspace_id, call_sid, step_id


def _prepare_callback_claim(
    workspace_id: str, call_sid: str, step_id: str, claimer: CallbackClaimer,
) -> PreparedCallbackClaim:
    """Validate a trusted claim boundary without consuming its key."""
    claim = getattr(claimer, "claim", None)
    if not callable(claim):
        raise ValueError("claimer must provide a trusted callable claim method")
    return claim, _key(workspace_id, call_sid, step_id)


def _commit_callback_claim(prepared: PreparedCallbackClaim) -> None:
    """Consume a previously validated claim, rejecting replays and bad adapters."""
    claim, key = prepared
    claimed = claim(*key)
    if claimed is False:
        raise CallbackReplayError("callback step already claimed")
    if claimed is not True:
        raise ValueError("claimer must return exactly True or False")


class InMemoryCallbackClaimStore:
    """Thread-safe, bounded callback claim test double.

    Derive step_id from trusted, persisted call state after authenticating and
    binding a callback. The first claim for an exact workspace/call/step key
    returns True; later claims return False and must not repeat side effects.

    This store is intentionally process-local and never expires claims. It is
    useful for synthetic concurrency tests only, not distributed Lambda replay
    protection. It performs no authentication, authorization, persistence,
    logging, network access, or callback payload storage.
    """

    def __init__(self, *, max_entries: int = 1000) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._claims: set[tuple[str, str, str]] = set()
        self._lock = Lock()

    def claim(self, workspace_id: str, call_sid: str, step_id: str) -> bool:
        """Atomically claim one exact synthetic step, returning False on replay.

        Existing claims remain readable when the store is full. A new key at
        capacity raises CallbackClaimCapacityError without evicting old claims.
        """
        key = _key(workspace_id, call_sid, step_id)
        with self._lock:
            if key in self._claims:
                return False
            if len(self._claims) >= self._max_entries:
                raise CallbackClaimCapacityError("callback claim capacity reached")
            self._claims.add(key)
            return True


def validate_and_claim_call_event(
    event: Mapping[str, Any], *, public_url: str, validator: SignatureValidator,
    expected_account_sid: str, expected_call_sid: str, workspace_id: str,
    step_id: str, claimer: CallbackClaimer,
) -> dict[str, str]:
    """Authenticate, bind and atomically claim one trusted synthetic step.

    Workspace, expected call and step identifiers must come from trusted
    session state. Callback fields cannot select the claim key. Invalid trusted
    claim configuration is rejected before signature validation, while invalid
    transport, signature or call binding is rejected before the claimer runs.

    A successful claim returns the already validated form-field snapshot for
    subsequent local policy evaluation. An exact replay raises
    CallbackReplayError. Storage failures and capacity errors propagate so an
    adapter fails closed. This helper does not authorize a workspace, persist
    state, acknowledge a webhook or execute an effect.
    """
    prepared = _prepare_callback_claim(
        workspace_id, expected_call_sid, step_id, claimer,
    )
    fields = validate_call_event(
        event, public_url=public_url, validator=validator,
        expected_account_sid=expected_account_sid,
        expected_call_sid=expected_call_sid,
    )
    _commit_callback_claim(prepared)
    return fields
