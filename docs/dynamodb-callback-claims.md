# DynamoDB callback claim boundary

`zentomic.dynamodb_callback_claim_store.DynamoDBCallbackClaimStore` implements
the existing `CallbackClaimer` protocol. Importing or constructing it performs
no I/O. It requires a trusted low-level DynamoDB client with `put_item`, not a
boto3 `Table`. No SDK dependency, environment lookup, provisioning, migration,
or production wiring is supplied. Local tests use synthetic clients only.

Pass `client` and `table_name` as keyword arguments. The dedicated, existing
table must have string partition key `workspace_id` and string sort key
`callback_key`. Do not reuse the lifecycle table or unrelated production data.

The workspace remains an exact opaque key. The sort key is the call SID's UTF-8
byte length, a colon, the exact call SID, then the exact step ID. For example,
`call` and `step` produce `4:callstep`. Length-prefixing prevents delimiter
collisions without hashing or normalizing identifiers. Existing claim rules
limit each identifier to 256 UTF-8 bytes, so the compound key is at most 516
UTF-8 bytes. Only keys are stored, never callback fields or payloads.

Each `claim` sends one conditional
[PutItem](https://docs.aws.amazon.com/amazondynamodb/latest/APIReference/API_PutItem.html)
with `attribute_not_exists` checks on both key attributes. There is no
read-before-write race. A successful mapping response returns `True`; only a
client `ConditionalCheckFailedException` returns `False`. Existing rows are
never replaced, and no old item is requested on success or conditional failure.
Other client errors propagate unchanged. Malformed non-mapping responses raise
a generic `ValueError`. There are no adapter-level retries.

An optional trusted `before_request` hook runs after key validation and before
every put, including replay attempts. It must raise to deny work; its return
value is ignored. Hook failures propagate outside conditional-error translation.
It does not configure SDK timeouts, constrain SDK-internal retries, cancel a
request, or reserve time. Configure bounded SDK retries/timeouts separately.

Use the existing `validate_and_claim_call_event` or claimed voice helpers to
authenticate and bind the account/call before accessing the store. Authorize
the workspace and select the current step from trusted persisted state, never
callback fields. Direct `claim` calls do no authentication or authorization.

## Failure and retention boundaries

A timeout or malformed response may follow a saved claim. A later authenticated
delivery then returns `False`, including when the earlier handler never executed
its effect. This is at-most-once admission, not exactly-once delivery, a work
queue, a completion receipt, or a transaction with session state. Do not repeat
effects or delete claims to recover from uncertainty. Workflows needing atomic
claim/state transitions or reliable effect delivery need a separate reviewed
transaction/outbox and reconciliation design.

Claims do not expire here. Do not delete or TTL-expire rows while their callbacks
can replay. Supply retention policy, least-privilege permissions and workspace
authorization separately. Use a single-region writer; this adapter makes no
global-table concurrency guarantee. No storage capacity or cost bound is enforced.

Run `python -m tests tests.test_dynamodb_callback_claim_store` for synthetic
coverage of exact request shapes, key isolation, replay denial, input validation,
admission, uncertain write results, and authenticated REST v1/HTTP v2 composition
with plain/base64 bodies. These are not SDK, service, IAM, Twilio cryptography,
or live distributed-concurrency tests. No real client is invoked.

Run `python -m tests tests.test_dynamodb_claimed_voice_flow` for composition
regressions across the claimed keypad, speech, confirmation, and dial-result
helpers. Each delivery constructs a fresh adapter over the same synthetic
table. Changed authenticated input cannot reopen a claimed step; lost write
acknowledgments and policy failures after claiming also leave it consumed.
Neither an uncertain claim nor a replay reaches voice policy. Rejected
signatures, wrong account/parent/child identities, and denied request admission
leave the claim available. Coverage includes REST v1/HTTP v2 and plain/base64
bodies. These sequential fake-client checks demonstrate the composition's
failure boundaries, not real provider cryptography, distributed atomicity,
session transactions, or permission to execute a proposed effect.
