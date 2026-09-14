"""Process-local callback claims for offline adapter development, not Lambda."""

from threading import Lock


MAX_CLAIM_IDENTIFIER_BYTES = 256
_IDENTIFIER_ERROR = (
    "claim identifiers must be nonblank UTF-8 strings of at most 256 bytes"
)


class CallbackClaimCapacityError(ValueError):
    """A new callback cannot be claimed without exceeding this instance's limit."""


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
