"""Injectable DynamoDB lifecycle storage; importing it creates no AWS client."""

import re
from collections.abc import Mapping
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


def _snapshot(item: Any) -> CallStatusSnapshot:
    if not isinstance(item, Mapping):
        raise ValueError(_INVALID_SNAPSHOT)
    status = item.get("call_status")
    revision = item.get("status_revision")
    if (not isinstance(status, Mapping) or set(status) != {"S"}
            or not isinstance(revision, Mapping) or set(revision) != {"N"}):
        raise ValueError(_INVALID_SNAPSHOT)
    try:
        is_terminal_call_status(status["S"])
        number = revision["N"]
        if (not isinstance(number, str) or not 1 <= len(number) <= 128
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
    """

    def __init__(self, *, client: DynamoDBStatusClient, table_name: str) -> None:
        if (not isinstance(table_name, str) or not 3 <= len(table_name) <= 255
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", table_name)):
            raise ValueError("table_name must be a DynamoDB table name, not an ARN")
        if not all(callable(getattr(client, method, None))
                   for method in ("get_item", "update_item")):
            raise ValueError("client must provide callable get_item and update_item")
        self._client = client
        self._table_name = table_name

    def load(self, workspace_id: str, call_sid: str) -> CallStatusSnapshot:
        """Read current state consistently; only an absent Item means revision zero."""
        workspace_id, call_sid = _key(workspace_id, call_sid)
        key = {"workspace_id": {"S": workspace_id}, "call_sid": {"S": call_sid}}
        result = self._client.get_item(
            TableName=self._table_name, Key=key, ConsistentRead=True,
            ProjectionExpression="#workspace, #call, #status, #revision",
            ExpressionAttributeNames={
                "#workspace": "workspace_id", "#call": "call_sid",
                "#status": "call_status", "#revision": "status_revision",
            },
        )
        if not isinstance(result, Mapping):
            raise ValueError(_INVALID_SNAPSHOT)
        if "Item" not in result:
            return CallStatusSnapshot()
        item = result["Item"]
        if (not isinstance(item, Mapping)
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
            details = response.get("Error") if isinstance(response, Mapping) else None
            if (isinstance(details, Mapping)
                    and details.get("Code") == "ConditionalCheckFailedException"):
                raise StatusConflictError("call status revision conflict") from None
            raise
        if (not isinstance(result, Mapping)
                or _snapshot(result.get("Attributes")) != proposed):
            raise ValueError(_INVALID_SNAPSHOT)
        return proposed
