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

Run `python -m tests tests.test_dynamodb_status_isolation` for synthetic key-
isolation coverage: shared call IDs across workspaces, distinct call legs,
delimiter collisions, whitespace, and canonically equivalent Unicode must not
alias. Exact 256-byte UTF-8 key boundaries are preserved. Invalid keys fail
before admission or client access; a read returning another workspace/call's
row fails closed without writing. These tests exercise the adapter boundary,
not IAM enforcement, workspace authorization, or a live DynamoDB service.

Run `python -m tests tests.test_dynamodb_callback_authentication` to exercise
authentication through this adapter with synthetic REST v1/HTTP v2 callbacks,
both plain and base64 encoded. Rejected signatures, wrong account/call bindings,
and invalid lifecycle fields must fail before request admission or storage,
including when a terminal row already exists and after a conditional conflict.
Signed extra fields cannot replace trusted workspace keys or revisions, and an
injected validator cannot mutate the fields later persisted. The validator is a
test double: this does not verify Twilio cryptography or authorize workspaces.

`load` sends a [GetItem](https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_GetItem.html)
request with `ConsistentRead=True` and projects only these four attributes. Only
an absent `Item` means `CallStatusSnapshot(None, 0)`. A present but incomplete,
unknown-status, wrong-key, or invalid-revision row raises a generic `ValueError`.
Revisions must be positive integral numbers no larger than `10**38 - 1`; an
advance at that limit fails rather than wrapping. Valid integral decimal or
exponent representations are decoded without floating-point rounding.

Conditional revision comparisons are numeric: a stored `{"N": "1.0"}` must
match an expected `{"N": "1"}`. The synthetic client models this equivalence,
including during no-op writes. Run
`python -m tests tests.test_dynamodb_revision_numbers` for alternate encodings,
exact adjacent 38-digit revisions under reduced Decimal precision, and concurrent
status/revision changes that must still conflict. These are offline contract
tests, not service-level validation of DynamoDB number parsing or expressions.

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

### Fresh admission before each request

Optionally pass a trusted `before_request` callable. Construction validates but
does not invoke it. Every `load` invokes it immediately before `get_item`; every
`observe` invokes it before its reload and again before `update_item`, even for
a no-op. The authenticated orchestration's initial load is gated too. This
allows a request to be denied when an earlier read or conflict backoff consumed
the remaining call or invocation budget.

The hook must **raise** to deny work; its return value is ignored. For a client
with a separately enforced request-time envelope, bind the fixed-duration hook:

```python
from zentomic.lambda_budget import make_lambda_request_admission_hook

# Integration sketch: all three inputs come from the trusted invocation/session.
before_request = make_lambda_request_admission_hook(
    context=context, deadline_ms=stored_deadline_ms, now_ms=trusted_now_ms,
    required_request_ms=1_000, reserve_ms=500,
)
```

The example's 1,000 ms is synthetic, not a recommended SDK configuration. Set
`required_request_ms` to cover the **complete enforced request envelope**,
including SDK retries. Admission requires that whole duration plus reserve to
fit both budgets. Insufficient time raises `RequestBudgetExhaustedError`, a
`TimeoutError`; clock failures raise safe `ValueError` diagnostics. Do not pass
`make_lambda_operation_timeout_resolver` directly: its `None` result is ignored.
Pass `before_request=before_request` when constructing the store. Hook exceptions
propagate unchanged, outside the client's conditional-conflict translation.
Omitting the hook preserves existing behavior. Authenticate before invoking the
store and use a new hook/store binding for each invocation, retaining the fixed
persisted call deadline. Run `python -m tests tests.test_lambda_request_admission`
for synthetic fixed-envelope boundaries and composition with both stores.

Admission is not timeout enforcement: no duration is applied to the client,
SDK-internal retries do not call the hook, and already-sent requests
are not cancelled or rolled back. Configure bounded client timeouts and retries
separately, allowing for all attempts and cleanup. No time is reserved durably.
Run `python -m tests tests.test_dynamodb_request_admission` for synthetic coverage,
including expiry between read/write and readmission after a conditional conflict.

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
