"""Injectable DynamoDB lifecycle storage; importing it creates no AWS client."""

import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from zentomic.call_status import advance_call_status, is_terminal_call_status
from zentomic.call_status_store import CallStatusSnapshot, StatusConflictError, _key


MAX_STATUS_REVISION = 10**38 - 1
_INVALID_SNAPSHOT = "DynamoDB returned an invalid call status snapshot"


class DynamoDBStatusClient(Protocol):
    """Low-level client methods, not the high-level boto3 Table interface."""

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        """Read one item using DynamoDB AttributeValue dictionaries."""

    def update_item(self, **kwargs: Any) -> dict[str, Any]:
        """Atomically check and update one item."""


def _plain_dictionary(value: Any) -> bool:
    """Return whether a low-level SDK mapping is inert before key lookup."""
    return type(value) is dict and all(type(key) is str for key in value)


def _snapshot(item: Any) -> CallStatusSnapshot:
    # Low-level botocore responses are plain dictionaries. Reject mapping
    # subclasses and non-plain keys before membership, iteration, lookup, or
    # equality can invoke behavior supplied by an injected client.
    if not _plain_dictionary(item):
        raise ValueError(_INVALID_SNAPSHOT)
    status = item.get("call_status")
    revision = item.get("status_revision")
    if (not _plain_dictionary(status) or set(status) != {"S"}
            or not _plain_dictionary(revision) or set(revision) != {"N"}):
        raise ValueError(_INVALID_SNAPSHOT)
    try:
        is_terminal_call_status(status["S"])
        number = revision["N"]
        if (type(number) is not str or not 1 <= len(number) <= 128
                or not re.fullmatch(
                    r"\+?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
                    number,
                )):
            raise ValueError(_INVALID_SNAPSHOT)
        decoded = Decimal(number)
        if (not decoded.is_finite() or not 1 <= decoded <= MAX_STATUS_REVISION
                or decoded != decoded.to_integral_value()):
            raise ValueError(_INVALID_SNAPSHOT)
    except (ValueError, InvalidOperation):
        raise ValueError(_INVALID_SNAPSHOT) from None
    return CallStatusSnapshot(status["S"], int(decoded))


class DynamoDBCallStatusStore:
    """Conditional lifecycle writes through a trusted, injected low-level client.

    Requires a dedicated, existing table with string partition key workspace_id
    and string sort key call_sid. No table is created or inspected. Construction
    performs no I/O; load/observe do invoke the injected client. Tests must use a
    fake. This stores status/revision only, not authorization, deadlines, claims,
    retention or side effects. All writers must preserve this revision protocol.
    Use a single-region writer and retain terminal rows throughout callback replay.
    Configure the client's timeouts/retries outside this adapter.

    Optional trusted before_request runs immediately before each get/update.
    It must raise to deny work; its return value is ignored. Use a fresh hook
    per invocation to recheck call/runtime budgets after reads and backoff.
    Hook failures propagate unchanged. It does not bound SDK-internal retries,
    configure timeouts, reserve time, or undo a request already sent.
    """

    def __init__(
        self, *, client: DynamoDBStatusClient, table_name: str,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        # Configuration must be inert before length or regex operations can
        # execute behavior supplied by a string subclass.
        if (type(table_name) is not str or not 3 <= len(table_name) <= 255
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", table_name)):
            raise ValueError("table_name must be a DynamoDB table name, not an ARN")
        if not all(callable(getattr(client, method, None))
                   for method in ("get_item", "update_item")):
            raise ValueError("client must provide callable get_item and update_item")
        if before_request is not None and not callable(before_request):
            raise ValueError("before_request must be a trusted callable or None")
        self._client = client
        self._table_name = table_name
        self._before_request = before_request

    def load(self, workspace_id: str, call_sid: str) -> CallStatusSnapshot:
        """Read current state consistently; only an absent Item means revision zero."""
        workspace_id, call_sid = _key(workspace_id, call_sid)
        key = {"workspace_id": {"S": workspace_id}, "call_sid": {"S": call_sid}}
        if self._before_request is not None:
            self._before_request()
        result = self._client.get_item(
            TableName=self._table_name, Key=key, ConsistentRead=True,
            ProjectionExpression="#workspace, #call, #status, #revision",
            ExpressionAttributeNames={
                "#workspace": "workspace_id", "#call": "call_sid",
                "#status": "call_status", "#revision": "status_revision",
            },
        )
        if not _plain_dictionary(result):
            raise ValueError(_INVALID_SNAPSHOT)
        if "Item" not in result:
            return CallStatusSnapshot()
        item = result["Item"]
        if (not _plain_dictionary(item)
                or any(item.get(name) != value for name, value in key.items())):
            raise ValueError(_INVALID_SNAPSHOT)
        return _snapshot(item)

    def observe(
        self, workspace_id: str, call_sid: str, incoming_status: str,
        *, expected_revision: int,
    ) -> CallStatusSnapshot:
        """Reload, compute monotonically, then compare-and-set even for a no-op.

        Creation requires an absent row, not merely an absent revision. Updates
        check both loaded status and revision. Only a conditional-check failure
        becomes StatusConflictError; other client failures propagate. A successful
        write may precede a transport/response failure, so do not assume rollback.
        """
        workspace_id, call_sid = _key(workspace_id, call_sid)
        if (type(expected_revision) is not int
                or not 0 <= expected_revision <= MAX_STATUS_REVISION):
            raise ValueError("expected_revision must be an integer from 0 to 10**38 - 1")
        is_terminal_call_status(incoming_status)
        current = self.load(workspace_id, call_sid)
        if current.revision != expected_revision:
            raise StatusConflictError("call status revision conflict")
        status = advance_call_status(current.status, incoming_status)
        revision = current.revision + (status != current.status)
        if revision > MAX_STATUS_REVISION:
            raise ValueError("call status revision limit reached")
        proposed = CallStatusSnapshot(status, revision)
        names = {"#status": "call_status", "#revision": "status_revision"}
        values = {":status": {"S": status}, ":revision": {"N": str(revision)}}
        if current.revision == 0:
            names.update({"#workspace": "workspace_id", "#call": "call_sid"})
            condition = "attribute_not_exists(#workspace) AND attribute_not_exists(#call)"
        else:
            condition = "#revision = :expected_revision AND #status = :expected_status"
            values.update({
                ":expected_revision": {"N": str(current.revision)},
                ":expected_status": {"S": current.status},
            })
        # Admission failures are not DynamoDB conditional-check failures. Keep
        # the hook outside the client exception translation, including no-ops.
        if self._before_request is not None:
            self._before_request()
        try:
            result = self._client.update_item(
                TableName=self._table_name,
                Key={"workspace_id": {"S": workspace_id}, "call_sid": {"S": call_sid}},
                UpdateExpression="SET #status = :status, #revision = :revision",
                ConditionExpression=condition, ExpressionAttributeNames=names,
                ExpressionAttributeValues=values, ReturnValues="UPDATED_NEW",
            )
        except Exception as error:
            response = getattr(error, "response", None)
            # Botocore supplies plain dictionaries here. Reject programmatic
            # mapping, key, and string subclasses before invoking ``get`` or
            # equality hooks that could disguise a failure as a write conflict.
            details = response.get("Error") if _plain_dictionary(response) else None
            code = details.get("Code") if _plain_dictionary(details) else None
            if type(code) is str and code == "ConditionalCheckFailedException":
                raise StatusConflictError("call status revision conflict") from None
            raise
        if (not _plain_dictionary(result)
                or _snapshot(result.get("Attributes")) != proposed):
            raise ValueError(_INVALID_SNAPSHOT)
        return proposed
