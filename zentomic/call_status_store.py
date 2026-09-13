"""Process-local lifecycle storage for offline adapter development, not Lambda."""

from dataclasses import dataclass
from threading import Lock

from zentomic.call_status import advance_call_status, is_terminal_call_status


@dataclass(frozen=True)
class CallStatusSnapshot:
    """Immutable observation and revision; zero means no stored observation."""

    status: str | None = None
    revision: int = 0


class StatusConflictError(ValueError):
    """The caller must reload state and recompute against the latest revision."""


class StatusCapacityError(ValueError):
    """A new call cannot be stored without exceeding this instance's limit."""


def _key(workspace_id: str, call_sid: str) -> tuple[str, str]:
    for value in (workspace_id, call_sid):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("status identifiers must be nonblank UTF-8 strings")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("status identifiers must be nonblank UTF-8 strings") from None
    # Preserve exact opaque identifiers; concatenating could alias distinct keys.
    return workspace_id, call_sid


class InMemoryCallStatusStore:
    """Thread-safe test double for conditional, workspace/call-scoped writes.

    Authenticate, authorize the workspace, and bind the account/call before
    passing an observation here. This store does none of those checks. It only
    holds lifecycle status, not voice steps, deadlines, budgets or claims.

    The lock coordinates threads sharing this instance, not separate processes
    or Lambda invocations. State is not durable or TTL bounded; max_entries
    limits stored workspace/call pairs without eviction. Create short-lived
    instances for synthetic local tests only. No network,
    files, credentials, SDKs, callback deduplication or side effects are used.
    """

    def __init__(self, *, max_entries: int = 1000) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._states: dict[tuple[str, str], CallStatusSnapshot] = {}
        self._lock = Lock()

    def load(self, workspace_id: str, call_sid: str) -> CallStatusSnapshot:
        """Return an immutable snapshot; reading a missing key does not store it."""
        key = _key(workspace_id, call_sid)
        with self._lock:
            return self._states.get(key, CallStatusSnapshot())

    def observe(
        self, workspace_id: str, call_sid: str, incoming_status: str,
        *, expected_revision: int,
    ) -> CallStatusSnapshot:
        """Apply the monotonic policy only if the loaded revision still matches.

        A changed status increments once; duplicate/delayed observations keep
        the revision. A stale revision conflicts even for a no-op. Reload and
        recompute on conflict, never blindly overwrite or reset the revision.
        Success (including a terminal snapshot) does not authorize cleanup.
        At capacity, new keys raise StatusCapacityError; existing keys retain
        normal revision/transition semantics. Never evict terminal observations.
        """
        key = _key(workspace_id, call_sid)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        is_terminal_call_status(incoming_status)
        with self._lock:
            current = self._states.get(key, CallStatusSnapshot())
            if current.revision != expected_revision:
                raise StatusConflictError("call status revision conflict")
            status = advance_call_status(current.status, incoming_status)
            if status == current.status:
                return current
            if key not in self._states and len(self._states) >= self._max_entries:
                raise StatusCapacityError("call status capacity reached")
            updated = CallStatusSnapshot(status, current.revision + 1)
            self._states[key] = updated
            return updated
