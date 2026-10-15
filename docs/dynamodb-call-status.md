# DynamoDB lifecycle storage boundary

`DynamoDBCallStatusStore` implements the existing `CallStatusStore` protocol for
one call leg's monotonic status and revision. It does not implement a complete
call-session store. Importing or constructing it creates no SDK client and
performs no network calls. Calling `load` or `observe` invokes the client supplied
by the application, so use synthetic clients for local tests.

## Integration contract

Supply a trusted, already configured **low-level** DynamoDB client and an existing
table's name as keyword arguments `client` and `table_name`. The high-level boto3
`Table` interface is not compatible: this adapter uses DynamoDB AttributeValue
dictionaries such as `{"S": "demo-workspace"}` and `{"N": "1"}`. Table ARNs are not
accepted. There is no boto3 dependency, environment lookup, client construction,
table discovery, provisioning, credential configuration, or production wiring.

The dedicated table must have this exact schema. It is not a migration for an
existing Zentomic production table and must not be pointed at unrelated data.

| Attribute | Role | DynamoDB type |
| --- | --- | --- |
| `workspace_id` | Partition key | String |
| `call_sid` | Sort key | String |
| `call_status` | Stored lifecycle status | String |
| `status_revision` | Positive integral revision | Number |

Identifiers follow the existing store's exact, nonblank UTF-8, 256-byte-per-key
contract. They are not joined into a compound string or normalized. Authorization
and account/call binding must happen before storage access. Do not use callback
fields as trusted workspace IDs or expected revisions.

`load` sends a [GetItem](https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_GetItem.html)
request with `ConsistentRead=True` and projects only these four attributes. Only
an absent `Item` means `CallStatusSnapshot(None, 0)`. A present but incomplete,
unknown-status, wrong-key, or invalid-revision row raises a generic `ValueError`.
Revisions must be positive integral numbers no larger than `10**38 - 1`; an
advance at that limit fails rather than wrapping. Valid integral decimal or
exponent representations are decoded without floating-point rounding.

`observe` validates inputs, reloads the snapshot, checks `expected_revision`, and
applies the shared `advance_call_status` policy. It then sends an
[UpdateItem](https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_UpdateItem.html)
request that sets only status and revision:

- Creation requires both key attributes to be absent, not just a missing revision.
- Existing rows must still match both the loaded status and revision.
- No-op observations also perform the conditional write, preserving the revision.
- The first terminal outcome cannot be replaced by a delayed observation.
- Unrelated attributes are not replaced. Updated return values are checked against
  the proposed snapshot, not trusted without validation.

The [condition expression](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html)
protects the write after the read; the read alone does not prevent a race. All
writers must use the same revision protocol. A stale read or a write-time
`ConditionalCheckFailedException` raises `StatusConflictError`, compatible with
`retry_call_status_event`. Other client failures propagate unchanged. Malformed
write responses fail closed, but a transport or response error does **not** prove
that the write was rolled back. Reload and reconcile before retrying uncertain
operations; never treat them as permission to repeat an external side effect.

## Recovering an unacknowledged write

A timeout or invalid update response can occur **after** DynamoDB saved the
observation. `retry_call_status_event` does not classify either failure as a
conditional conflict and does not invoke its conflict-backoff hook for them.
Do not catch arbitrary storage errors and convert them to `StatusConflictError`.

On a later delivery, authenticate and bind the callback again, reload the current
snapshot, and recompute against that revision. Do not reuse the failed attempt's
expected revision or infer absence from its exception. Reobserving an already
saved status keeps its revision; an intervening terminal observation still wins
over delayed active or different terminal statuses. No-op observations still
need a conditional write and can themselves fail or conflict.

This reconciles lifecycle state only. It is not exactly-once processing or a
receipt for calls, messages, cleanup, or acknowledgments. Those effects still
need their own durable idempotency protocol and adapter-specific error policy.

## Boundaries before any live use

This adapter can issue service requests when given a real client. No real client
is instantiated or invoked by the repository examples or tests. The application
must separately provide least-privilege access, workspace authorization, trusted
configuration, deployment review, and live integration validation.

Configure and enforce client connection/read timeouts, SDK retry limits, and
whole-call/invocation budgets outside this adapter. One `observe` performs a read
and a conditional write, even for a no-op. An authenticated orchestration attempt
also loads the store first. Budget for **all** requests and retries; the adapter
does not promise an operation timeout or a cost ceiling. Use a single-region
writer; no cross-region/global-table concurrency guarantee is provided here.

Do not delete or TTL-expire terminal rows while callbacks may replay: a missing
row permits creation again. This adapter does not supply retention, durable
callback claims, session deadlines, voice steps, account IDs, cleanup claims,
transactional side effects, or response acknowledgments. It never authorizes a
real call, message, cleanup action, or customer-data disclosure.

Run `python -m tests tests.test_dynamodb_call_status_store` for the focused suite.
Tests assert exact low-level request shapes and compare all 512 three-observation
sequences with the in-memory policy. A small synthetic client models competing
creation, terminal updates, no-op races, and deletion, and exercises the complete
authenticated conflict retry. It is not a DynamoDB emulator or proof of service
expression evaluation. No DynamoDB Local, AWS, boto3, real signature validator,
credentials, table, or network integration was used. Validate those separately
before connecting any production traffic.

Run `python -m tests tests.test_dynamodb_status_recovery` for committed-but-
unacknowledged write scenarios. Its synthetic client saves the row and then
raises a timeout or returns a malformed response. The tests verify no automatic
conflict retry, authenticated redelivery, stable revisions, intervening terminal
state preservation, and no storage access after failed reauthentication. This
models an uncertain response; it does not simulate SDK retries or service faults.
