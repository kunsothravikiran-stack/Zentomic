"""Injectable conditional callback claims; importing creates no AWS client."""

import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from zentomic.callback_claim_store import _key


class DynamoDBClaimClient(Protocol):
    """Low-level client boundary, not the boto3 Table interface."""

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        """Conditionally create an item using AttributeValue dictionaries."""


class DynamoDBCallbackClaimStore:
    """Implement CallbackClaimer with one conditional put per trusted key.

    Requires an existing, dedicated table with string workspace_id partition key
    and string callback_key sort key. Construction performs no I/O. Calls use
    the injected client, so use a fake in local tests. Claims never expire here.
    Authorize the workspace, authenticate and bind the call before claiming.

    This is an at-most-once admission marker, not a transaction with session
    state or an exactly-once effect receipt. Failures can follow a saved claim;
    never delete it or assume a failed request permits executing the effect.
    Use a single-region writer and configure SDK timeouts/retries externally.
    """

    def __init__(
        self, *, client: DynamoDBClaimClient, table_name: str,
        before_request: Callable[[], None] | None = None,
    ) -> None:
        if (not isinstance(table_name, str) or not 3 <= len(table_name) <= 255
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", table_name)):
            raise ValueError("table_name must be a DynamoDB table name, not an ARN")
        if not callable(getattr(client, "put_item", None)):
            raise ValueError("client must provide callable put_item")
        if before_request is not None and not callable(before_request):
            raise ValueError("before_request must be a trusted callable or None")
        self._client = client
        self._table_name = table_name
        self._before_request = before_request

    def claim(self, workspace_id: str, call_sid: str, step_id: str) -> bool:
        """Return True on creation, False only on a conditional-check failure.

        Preserve exact identifiers with a UTF-8-byte-length-prefixed call SID.
        Unlike delimiter joining, this cannot alias different call/step pairs.
        Invalid inputs fail before admission or client access. The optional
        trusted hook must raise to deny work; its return value is ignored.
        """
        workspace_id, call_sid, step_id = _key(workspace_id, call_sid, step_id)
        callback_key = f"{len(call_sid.encode('utf-8'))}:{call_sid}{step_id}"
        if self._before_request is not None:
            self._before_request()
        try:
            result = self._client.put_item(
                TableName=self._table_name,
                Item={
                    "workspace_id": {"S": workspace_id},
                    "callback_key": {"S": callback_key},
                },
                ConditionExpression=(
                    "attribute_not_exists(#workspace) AND "
                    "attribute_not_exists(#callback)"
                ),
                ExpressionAttributeNames={
                    "#workspace": "workspace_id", "#callback": "callback_key",
                },
                ReturnValues="NONE", ReturnValuesOnConditionCheckFailure="NONE",
            )
        except Exception as error:
            response = getattr(error, "response", None)
            details = response.get("Error") if isinstance(response, Mapping) else None
            if (isinstance(details, Mapping)
                    and details.get("Code") == "ConditionalCheckFailedException"):
                return False
            raise
        if not isinstance(result, Mapping):
            raise ValueError("DynamoDB returned an invalid callback claim response")
        return True
