# Zentomic

Modern business IVR and AI call routing, intended for Python AWS Lambda,
API Gateway, DynamoDB, Twilio, and OpenAI.

## Current scope

This repository starts with an offline, dependency-free Python scaffold. It is
not an export of an existing production service. There are no cloud resources,
deployment workflows, credentials, telephony operations, or AI API calls here.

Implemented: a Lambda-compatible `/health` liveness endpoint supporting GET and
HEAD with API Gateway REST API (v1) and HTTP API (v2) proxy events. GET returns
`{"status":"ok","service":"zentomic"}` without checking external services.
HEAD supports body-free liveness probes with the same status and headers as GET,
following [HTTP HEAD semantics](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.3.2).
Its proxy `body` is an empty string, including for unknown or invalid paths in
supported proxy formats. Other paths return 404; other methods on `/health`
return 405 with `Allow: GET, HEAD`. Method names are case-sensitive.
Unsupported event versions return 400. Request contents are not logged or
reflected in responses.

Also implemented: a pure single-digit IVR menu resolver. It selects an opaque
target identifier for digits `0` through `9`, with an explicit fallback for
missing, malformed, or unmapped input. It does not dial or expose a new endpoint.

```python
from zentomic.routing import resolve_dtmf

target = resolve_dtmf(
    "2",
    {"0": "reception", "1": "sales", "2": "support"},
    fallback_target="reception",
)
assert target == "support"
```

Menus may be empty or contain up to ten single ASCII digit keys. Targets and
the required fallback must be nonblank UTF-8 encodable strings; invalid configuration raises
`ValueError` before selecting a route. Input is not trimmed or coerced, so
multi-digit values, whitespace, Unicode digits, `*`, and `#` use the fallback.
The resolver does not log input, mutate the menu, or access the environment or
network. Identifiers are returned unchanged. A future integration must load an
authorized workspace's menu and resolve its targets within that same workspace;
this helper does not perform authentication or workspace authorization.

Target validation is shared by keypad, confirmed-intent, and dial-result
policies, including their collection and speech helpers. Surrogate code points
are rejected before a decision, even in an unused route or a fallback that
would not be selected. This avoids carrying an identifier that cannot be
encoded for future session persistence. Valid Unicode identifiers, including
their whitespace and normalization form, are preserved exactly. This is not
database-key validation, persistence, or workspace authorization.

### Bounded input retries

`resolve_gather` wraps the single-digit resolver with a finite collection
budget. This lets a future IVR adapter reprompt on silence or invalid digits
without trapping callers in an endless menu:

```python
from zentomic.gather import resolve_gather

decision = resolve_gather(
    None,
    {"0": "reception", "1": "sales"},
    fallback_target="reception",
    attempts=0,
    max_attempts=3,
)
assert (decision.action, decision.target, decision.attempts) == ("retry", None, 1)
```

`attempts` counts completed collections before this input, including silence;
the immutable result includes the updated count. Configured digits return
`route`, including a key explicitly mapped to reception. Missing, malformed,
or unmapped digits return `retry` with no target until the last allowed attempt,
then `fallback`. Valid input on the last attempt still routes. An already
exhausted budget always falls back and leaves the count unchanged. Counts must
be nonnegative integers and the maximum must be a positive integer (default 3);
booleans are rejected. All menu configuration is validated even after exhaustion.

An optional `hangup_digit` lets callers explicitly end an active menu:

```python
from zentomic.gather import resolve_gather

decision = resolve_gather(
    "9", {"0": "reception", "1": "sales"},
    fallback_target="reception", attempts=0, hangup_digit="9",
)
assert (decision.action, decision.target, decision.attempts) == ("hangup", None, 1)
```

The default `None` leaves all existing routing behavior unchanged. A configured
hangup key must be an unused single ASCII digit from `0` through `9`; overlap
with a route is an error, not a precedence rule. Input matches exactly without
normalization. Hangup consumes one attempt, including on the last allowed
collection. An already exhausted budget still falls back without incrementing
or acting on the digit. Invalid configuration is rejected in either case.
Use trusted menu configuration and mention the exit key in the collection
prompt. For `hangup`, a future adapter must persist the terminal decision and
return `render_hangup()` (described below), never reprompt or dial the fallback.
This decision alone does not end a real call or provide replay protection.

This helper does not expose an endpoint, generate TwiML, or persist state.
Before using it with real traffic, an adapter must authenticate callbacks,
atomically deduplicate them, and persist the returned count in trusted,
workspace-scoped call-session state. Do not accept attempt counts from callers
or reset them on each callback. Transport errors are not collection attempts.

### Whole-call deadline admission

Per-step retries do not bound the total time spent across speech, confirmation,
and forwarding. `resolve_call_budget` checks one fixed deadline shared by those
steps without reading a clock or contacting a service:

```python
from zentomic.budget import resolve_call_budget

# Synthetic millisecond timestamps in the same clock domain.
# A future adapter persists the deadline once when the call session starts.
decision = resolve_call_budget(deadline_ms=601_000, now_ms=501_000)
assert (decision.action, decision.remaining_ms) == ("continue", 100_000)
assert resolve_call_budget(deadline_ms=601_000, now_ms=601_000).action == "hangup"
```

At or after the deadline, the immutable result is `hangup` with zero remaining
milliseconds, never a human fallback that could prolong the session. Both inputs
must be nonnegative integers; booleans, floats, and strings are rejected. There
is no default duration or product-plan policy. Use a trusted current clock and
a deadline from authorized session state, never caller/model fields. Do not
reset the deadline on retries, step transitions, or fallback.

For a step that needs a minimum amount of time, supply a trusted
`minimum_remaining_ms` threshold, including any safety margin:

```python
from zentomic.budget import resolve_call_budget

decision = resolve_call_budget(
    deadline_ms=601_000, now_ms=598_000, minimum_remaining_ms=5_000,
)
assert (decision.action, decision.remaining_ms) == ("hangup", 3_000)
```

The threshold must be a positive integer, not a boolean, float, or request
field. Exactly enough time permits `continue`; less time selects `hangup`
even before expiry. The default of 1 millisecond preserves deadline-only
admission. `remaining_ms` always reports actual time left, so adapters must
check `action`, not just whether the remaining time is positive. This does not
reserve or deduct time, guarantee completion, or enforce an operation timeout.

A future adapter must check before admitting each operation and persist a
terminal outcome before returning `render_hangup()`. This is only admission:
it does not stop an active Gather, Dial, or model request, schedule a timer,
enforce a billing cap, or prevent replay. Operation-specific timeouts still
need to account for the remaining budget. Clock rollback can extend admission;
clock consistency and durable terminal-state handling remain adapter concerns.

For operations accepting a millisecond timeout, `resolve_operation_timeout`
caps the configured maximum against this same deadline, leaving an optional
cleanup/transport margin:

```python
from zentomic.budget import resolve_operation_timeout

timeout_ms = resolve_operation_timeout(
    deadline_ms=601_000, now_ms=598_000,
    maximum_ms=5_000, minimum_ms=1_000, reserve_ms=500,
)
assert timeout_ms == 2_500
```

It returns `None` when the remaining time minus the reserve is below the
minimum, including at expiry. Do not start the operation in that case; follow
the terminal path. The minimum and maximum must be positive integer
milliseconds with minimum <= maximum; the reserve must be a nonnegative
integer. Booleans are rejected. Defaults are a 1 ms minimum and no reserve.
Invalid limits raise `ValueError` even after expiry. Exact integer arithmetic
is used without rounding up or resetting the original deadline.

For an operation accepting only whole seconds, set `granularity_ms=1000`:

```python
from zentomic.budget import resolve_operation_timeout

timeout_ms = resolve_operation_timeout(
    deadline_ms=601_000, now_ms=598_000,
    maximum_ms=5_000, minimum_ms=1_000, reserve_ms=500,
    granularity_ms=1_000,
)
assert timeout_ms == 2_000
timeout_seconds = timeout_ms // 1_000  # Only convert after checking for None.
```

The positive-integer granularity defaults to 1 ms, preserving existing behavior.
The helper rounds down after applying both the maximum and reserve, then checks
the minimum again. It returns `None`, never zero, if no usable multiple fits,
even when unrounded time met the minimum. Limits need not be exact multiples.
The result remains milliseconds; granularity does not change the return unit.

For work that must also finish within the current serverless invocation, pass
`invocation_remaining_ms` from a fresh, trusted runtime remaining-time reading:

```python
from zentomic.budget import resolve_operation_timeout

timeout_ms = resolve_operation_timeout(
    deadline_ms=601_000, now_ms=598_000,
    maximum_ms=5_000, minimum_ms=1_000, reserve_ms=500,
    invocation_remaining_ms=2_000,  # Synthetic remaining duration, not a timestamp.
)
assert timeout_ms == 1_500
```

The smaller of call time and invocation time is used before deducting the
reserve, applying the operation maximum, and rounding down. The reserve is
deducted once, leaving that margin within both budgets. The optional value must
be a nonnegative integer; zero admits no work, and `None` (the default) leaves
existing behavior unchanged. Invalid values raise `ValueError` even after the
call expires. Refresh this reading before each operation or retry, never use
the invocation's original timeout or a caller field. A new invocation's larger
budget cannot extend the persisted call deadline. `None` means do not start the
operation, not that the runtime has enough time left to persist or send a
terminal response. This adds no runtime integration or cancellation mechanism.

The adapter must actually apply the returned timeout, recheck on retries, and
include any delay before execution in its margin. This pure helper neither
cancels work nor guarantees cleanup fits in the reserve. Convert units without
rounding up; if a provider's minimum or granularity cannot fit, skip the work.
It is not a direct Gather/Dial wall-clock guarantee: those verbs have separate
timing semantics. No SDK integration, timer, or active cancellation is added.

### Offline TwiML collection renderer

`render_dtmf_gather` serializes a single hosted menu response without contacting
Twilio or exposing an endpoint:

```python
from zentomic.twiml import render_dtmf_gather

xml = render_dtmf_gather(
    "Press 1 for sales. Press 0 for reception.",
    action_path="/voice/menu-result",
)
```

The generated `Gather` collects one keypad character, uses POST, and requests a
callback even on silence (`actionOnEmptyResult="true"`). No finish key is used,
so `*` and `#` can reach the resolver as invalid input. These attributes follow
the [Twilio Gather reference](https://www.twilio.com/docs/voice/twiml/gather).
Prompt markup is escaped as text, and invalid XML characters are rejected.
Project limits are 1000 prompt characters and a 1–60 second timeout (default 5).

Callback paths must contain slash-separated letters, ASCII digits, underscores,
or hyphens, starting with `/`. External URLs, queries, fragments, dot segments,
and percent escapes are intentionally unsupported. Use trusted configuration,
not caller/model input. This renderer is for hosted webhook responses only;
inline TwiML in the Calls API requires absolute callback URLs and is unsupported.

The example path is not implemented. A future adapter must authenticate the
callback, deduplicate it, and pass its input plus trusted attempt state to
`resolve_gather`. Render another menu only for `retry` (or initial collection),
never for `route` or `fallback`. Return XML with `Content-Type: application/xml`.
Serialization alone does not enforce retry budgets, authorize targets, or dial.

`render_hangup` provides a separate terminal response for a future explicit
end-call flow, with an optional farewell:

```python
from zentomic.twiml import render_hangup

xml = render_hangup("Thank you for calling. Goodbye.")
# <Response><Say>Thank you for calling. Goodbye.</Say><Hangup /></Response>
```

Omit the prompt (or pass `None`) for only `Hangup`. Supplied text uses the same
nonblank, 1000-character and XML-safety limits as collection prompts. The
attribute-free `Hangup` is always the final top-level verb, following the
[Twilio Hangup reference](https://www.twilio.com/docs/voice/twiml/hangup).
Unlike `Reject`, `Hangup` does not prevent answering a call or provider billing.
Rendering remains fully offline. This does not change routing: `fallback`
still selects its configured target, and no digit implicitly ends a call.
Only an explicitly configured `resolve_gather(..., hangup_digit=...)` opts
into the separate terminal decision described above.
A future authenticated adapter must decide when termination is appropriate;
the renderer neither verifies caller intent nor changes persisted session state.

### Offline speech collection renderer

`render_speech_gather` prepares an English, speech-only collection for a future
intent classifier, without starting a call or contacting a provider:

```python
from zentomic.twiml import render_speech_gather

xml = render_speech_gather(
    "How can we help you today?",
    action_path="/voice/speech-result",
    timeout=5,
    speech_timeout=2,
)
```

The XML uses `input="speech"`, `language="en-US"`, POST, and
`actionOnEmptyResult="true"`. `timeout` controls waiting for input;
`speech_timeout` controls the pause ending an utterance. Neither is a
whole-call duration or billing cap. These attributes follow the
[Twilio Gather reference](https://www.twilio.com/docs/voice/twiml/gather).
Both timeouts are project-limited to integers from 1 through 60 seconds;
booleans and `"auto"` are intentionally unsupported. Prompt and callback-path
validation are identical to the keypad renderer. No speech model or partial
transcription callback is selected; keypad collection remains unchanged.

The example endpoint is not implemented. A future adapter must authenticate
and bind callbacks to the current call step, deduplicate them, and handle
missing speech with a persisted retry budget. Treat `SpeechResult` as
untrusted text, not a route identifier or evidence of caller confirmation.
Classification must produce an allowlisted pending intent and require explicit
confirmation before forwarding. This renderer does not implement that flow,
transcription, storage, consent handling, or production speech integration.
Generating XML is offline; executing speech collection with a provider is not.

### Bounded speech result admission

`resolve_speech_gather` applies a finite collection budget before a future
classifier receives text. It does not call a model or select an intent:

```python
from zentomic.speech import resolve_speech_gather

decision = resolve_speech_gather(
    "I need billing help", fallback_target="reception", attempts=0,
)
assert (decision.action, decision.transcript, decision.attempts) == (
    "classify", "I need billing help", 1,
)
assert decision.target is None
```

Supply `SpeechResult` only after authenticating and binding the callback to
the current speech step. Missing, non-string, blank, oversized, or invalid
Unicode text retries within the budget, then falls back to the configured
human target. The fixed
`MAX_TRANSCRIPT_CHARS` limit is 2000 Unicode characters, including surrounding
whitespace, not bytes or model tokens. Accepted text is preserved unchanged.
Surrogate code points in Python strings are rejected without replacement or
normalization, so admitted text can be encoded for a classifier dependency.
The form decoder already rejects malformed UTF-8; this also protects direct
uses of the speech policy outside that transport decoder.
Every completed collection consumes one attempt. Valid text on the last
attempt can still be classified; an already exhausted budget always falls
back without incrementing. Counts and fallback configuration follow
`resolve_gather` validation. Retry/fallback decisions contain no transcript.

`SpeechDecision` omits its transcript from `repr` and `str`, including when a
decision appears inside a list or dictionary, to reduce accidental disclosure
in diagnostics. The original text remains available through `.transcript` for
classification and still participates in equality. This is not general-purpose
redaction: explicit attribute logging, `dataclasses.asdict`, and other object
serialization can still expose caller text. Do not log or persist those without
appropriate privacy controls.

`classify` is permission to proceed to a separately configured classifier,
not confirmation or authorization to dial. Text remains untrusted, including
instructions embedded in speech. Classifier output must still pass the
allowlisted intent and caller-confirmation policies. A future adapter must
atomically deduplicate and persist the decision before model calls or retries;
never take the attempt count from callback fields or reset it on each request.
This helper neither persists state nor enforces model costs or whole-call limits.

### Strict classifier response boundary

`parse_intent_response` admits a pending intent from a future classifier's text
response. The application contract is one JSON object with exactly one field:

```python
from zentomic.classification import parse_intent_response

pending = parse_intent_response(
    '{"intent":"sales"}', allowed_intents=("sales", "support"),
)
assert pending == "sales"
```

Labels match the trusted workspace allowlist exactly, without trimming or case
folding. Invalid output returns `None`: unknown/non-string labels, duplicate
JSON keys (including escaped duplicates), extra fields, Markdown wrappers,
malformed JSON, and responses over 4096 UTF-8 bytes including whitespace.
Model-supplied destinations, confirmation flags, and retry counts are never
accepted. Invalid allowlist configuration raises `ValueError`; an empty
allowlist admits nothing. Allowlist labels must be UTF-8 encodable, even when
the response is missing or invalid. Surrogate code points in configuration are
rejected with a generic error before parsing, so an escaped JSON string cannot
admit a non-encodable pending label. Valid Unicode labels and JSON surrogate
pairs representing real Unicode characters remain supported; labels are never
normalized, repaired, or case-folded. This check applies to classifier admission,
not to all identifiers in the routing helpers.
The parser does not log output or call a provider.

A future adapter must handle `None` with a bounded retry or human fallback,
persist accepted labels as pending within the authenticated current call step,
and use the separate caller-confirmation policy before routing. Acceptance is
not proof of classification accuracy, prompt-injection resistance, caller
confirmation, or workspace authorization. This adds no model/SDK integration,
provider response-envelope handling, endpoint, persistence, or token-cost cap.

### Hosted voice-step transitions

`resolve_transition_budget` bounds rapid step changes even when the whole-call
deadline has not expired. Pass the previously admitted transition count from
trusted call-session state and an explicit configured maximum:

```python
from zentomic.budget import resolve_transition_budget

decision = resolve_transition_budget(transitions=2, max_transitions=3)
assert (decision.action, decision.transitions) == ("continue", 3)
decision = resolve_transition_budget(transitions=decision.transitions, max_transitions=3)
assert (decision.action, decision.transitions) == ("hangup", 3)
```

Each `continue` admits exactly one next step, including the last allowed one.
At or above the limit, `hangup` preserves the count; a zero limit admits no
transitions. Both counts must be nonnegative integers, never booleans or caller
input. Authenticate, bind, and deduplicate callbacks first. Atomically persist
the new count and authorized next step before acting. Keep one counter across
menus, confirmation, and fallback; do not reset it on redirects or new Lambda
invocations. Replays and concurrent writes require a storage-level guard, which
this helper does not implement. Continue enforcing whole-call deadlines and
per-step retry budgets independently. On `hangup`, emit `render_hangup()` rather
than redirecting or dialing. Tests simulate a rapid redirect loop offline;
no live endpoint, persistence, or provider integration is added.

`render_redirect` serializes a server-selected transition, for example from an
accepted pending intent to its confirmation step. It emits only a top-level
[`Redirect` with explicit POST](https://www.twilio.com/docs/voice/twiml/redirect),
not an HTTP 3xx response or a phone transfer:

```python
from zentomic.twiml import render_redirect
from zentomic.response import twiml_response

response = twiml_response(render_redirect(action_path="/voice/confirm-intent"))
assert response["statusCode"] == 200
assert response["body"] == (
    '<Response><Redirect method="POST">/voice/confirm-intent</Redirect></Response>'
)
```

Paths use the existing root-relative named-segment policy: no external hosts,
queries, fragments, traversal, or encoded separators. Use trusted application
configuration, never caller/model input. Persist the authorized next step before
responding; authenticate and bind the next callback to that stored step. Carry
forward the original deadline and bounded transition/retry counters. Rendering
does not prevent loops, reset counters, fetch a URL, or add an endpoint. These
paths require hosted TwiML with a base URL, not inline Calls API TwiML.

### TwiML proxy responses

`twiml_response` wraps an existing renderer's output in a success envelope for
API Gateway [REST v1 proxy integrations](https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html)
and [HTTP v2 proxy integrations](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html).
It returns status 200, `Content-Type: application/xml; charset=utf-8`,
`Cache-Control: no-store`, and `isBase64Encoded: false`. The XML stays unchanged
in `body`, not JSON-quoted or base64-encoded. Return the dictionary from a future
Lambda adapter; let the runtime serialize the outer envelope.

```python
from zentomic.response import twiml_response
from zentomic.twiml import render_hangup

response = twiml_response(render_hangup("Goodbye."))
assert response["statusCode"] == 200
assert response["body"] == "<Response><Say>Goodbye.</Say><Hangup /></Response>"
assert response["isBase64Encoded"] is False
```

Supply only application-generated TwiML after authenticating the callback,
authorizing the decision, and persisting required call state. This helper checks
only that the input is a nonblank UTF-8 encodable string, not XML validity,
allowed verbs, or destination safety. It does not sanitize caller/model XML.
Authentication failures require a separate non-success response, not this
success-only wrapper. Headers are newly allocated per invocation; no request
headers are reflected. Offline tests cover all four renderers, Unicode and XML
escaping, JSON envelope round trips, invalid text, and independent headers.
This adds no voice endpoint or live API Gateway/provider verification; the
existing `/health` handler remains unchanged.

### Offline forwarding renderer

`render_dial(numbers, action_path=..., timeout=20, time_limit=14400)` serializes one `Dial` with
1-10 unique `Number` children. Supply a list or tuple of already authorized,
workspace-resolved phone destinations, not the opaque identifiers returned by
the routing helpers. Values must have E.164-style syntax: `+`, a nonzero first
digit, and 2-15 ASCII digits total. No trimming or normalization is performed;
extensions, SIP addresses, duplicates, and malformed values raise `ValueError`.
Syntax validation does not establish number assignment, ownership, or permission.

The renderer explicitly selects simultaneous ringing and disables Dial recording.
The first connected destination wins, which can include voicemail, as explained
in the [Twilio Number reference](https://www.twilio.com/docs/voice/twiml/number).
The required action uses POST and the same root-relative path restrictions as
the Gather renderer. A future action handler must authenticate and handle the
Dial outcome, including no-answer/busy/failure, before deciding what happens next.
The project accepts integer ringing timeouts of 5-60 seconds (default 20).
This is not a conversation-duration or billing cap; Twilio also adds a ringing
buffer. See the [Dial reference](https://www.twilio.com/docs/voice/twiml/dial).

Use `time_limit` to bound the connected duration of each Dial separately from
ringing. It emits Twilio's `timeLimit` attribute and accepts integers from 1 to
14400 seconds; booleans, floats, strings, and `None` are rejected. The default
explicitly preserves Twilio's standard four-hour limit. For example:

```python
from zentomic.twiml import render_dial

xml = render_dial(
    ["+12025550100"],  # Fictional fixture, not a real destination.
    action_path="/voice/dial-result", timeout=20, time_limit=600,
)
```

This requests a ten-minute connected limit for this forwarding step, following
the [Twilio timeLimit contract](https://www.twilio.com/docs/voice/twiml/dial#timelimit).
Longer account-specific limits are intentionally unsupported. Supply the limit
from trusted workspace policy, not caller/model input. It does not cap the
parent call, menu time, later fallback attempts, or charges. A future adapter
must track the remaining session budget across steps, terminate on exhaustion,
and handle the authenticated Dial action callback. Serialization alone does
not enforce a live limit or verify provider behavior.

This is XML serialization only: it does not send TwiML, expose a voice endpoint,
load numbers, or make calls. Use only trusted application configuration. A live
adapter must authorize each destination within the current workspace, enforce
dialing/cost policy, deduplicate callbacks, and manage call state. Never pass
caller/model-supplied phone numbers directly to this renderer. Output contains
destinations, so do not log it with real configuration. Hosted webhook responses
only; inline Calls API TwiML remains unsupported.

### Bounded forwarding outcomes

`resolve_dial_result` supplies an offline policy for the Number forwarding
renderer above:

```python
from zentomic.dial import resolve_dial_result

decision = resolve_dial_result(
    "no-answer", fallback_target="reception", fallback_used=False,
)
assert (decision.action, decision.target) == ("fallback", "reception")
```

Pass authenticated `DialCallStatus`, not the parent `CallStatus` or a Number
status-callback event. The supported values follow the
[Twilio Dial action contract](https://www.twilio.com/docs/voice/twiml/dial#dialcallstatus):
`busy`, `no-answer`, and `failed` select the configured fallback once;
`completed` and `canceled` select `hangup` with no target. This is Zentomic's
policy, not a provider-mandated next action. Once `fallback_used=True`, failures
also select `hangup`, preventing repeated forwarding through this policy.
Missing, malformed, unknown, and Conference-only `answered` statuses raise
`ValueError` without echoing input. Values are not normalized. The fallback
must be a nonblank opaque identifier and the state flag must be an actual
boolean, even for terminal outcomes. Decisions are immutable.

An adapter must authenticate and bind the callback to the current call and
expected dial step, authorize the fallback within that workspace, and atomically
deduplicate the callback and mark fallback as used **before** executing it.
Load this flag from trusted session state, never the callback; do not reset it
between attempts. These guarantees are not implemented by this pure helper.
Do not apply this Number-only policy to conferences or child status callbacks.
A `hangup` decision can use `render_hangup`; a `fallback` decision still needs
authorized target resolution. No endpoint, persistence, or real call is added.

### Call lifecycle status classification

`is_terminal_call_status` distinguishes a call leg's terminal outcomes from
known active states. This is separate from the forwarding fallback policy:

```python
from zentomic.call_status import is_terminal_call_status

assert is_terminal_call_status("completed") is True
assert is_terminal_call_status("no-answer") is True
assert is_terminal_call_status("in-progress") is False
```

Terminal values are `completed`, `busy`, `failed`, `no-answer`, and `canceled`;
`queued`, `ringing`, and `in-progress` return `False`. Missing, unknown, or
non-string values raise a generic `ValueError`. Matching is exact, without
normalization. These are
[Twilio Call Status values](https://www.twilio.com/docs/voice/api/call-resource#call-status-values),
not subscription event names such as `initiated` or `answered`. `completed`
does not prove a human answered or a business task succeeded.

Use authenticated `CallStatus` bound to the expected account and call leg,
not `DialCallStatus`. A child leg ending does not imply the parent call ended.
Twilio documents that status callbacks can arrive out of order. This helper
classifies only the supplied value; a future adapter must atomically deduplicate
updates and prevent delayed active events from reopening terminal state.
It does not authorize cleanup, persist state, end calls, expose an endpoint,
or make provider requests. Unknown statuses must not trigger cleanup or reopen
a session. Synthetic tests cover all values, strict rejection, and composition
with the injected authentication gate across both proxy formats and encodings;
they do not verify real signatures or delivery ordering.

`advance_call_status` adds a pure application policy for retaining progress:

```python
from zentomic.call_status import advance_call_status

assert advance_call_status(None, "ringing") == "ringing"
assert advance_call_status("in-progress", "queued") == "in-progress"
assert advance_call_status("ringing", "completed") == "completed"
assert advance_call_status("completed", "ringing") == "completed"
assert advance_call_status("completed", "failed") == "completed"
```

`None` denotes no stored observation. Active progress follows the project order
`queued` < `ringing` < `in-progress`, allowing missing intermediate events. The
first stored terminal outcome is retained, including for conflicting terminal
observations. This is an application retention policy, not reconstruction of
provider event order or reconciliation of conflicting outcomes. Both statuses
are strictly validated, even after termination; only the stored value may be
`None`. Invalid state is rejected without reflecting its contents.

Authenticate and bind the incoming callback to the same account and call leg
before using this helper. Read current state from trusted workspace-scoped
storage, then conditionally save against the version read. If another writer
wins, reload and recompute; an unconditional write can still lose terminal
state. Returning an unchanged or terminal value does not deduplicate delivery
or authorize repeated cleanup. Atomic storage, reconciliation, idempotent side
effects, and parent/child call coordination remain future adapter work.
Offline tests cover every known state pair and delayed/duplicate sequences.
`tests/test_call_status_flow.py` also composes transport decoding, mocked
signature verification, call binding, and lifecycle transitions across both
proxy formats and body encodings. It checks that rejected callbacks leave the
local state unchanged and signature failures never reach the transition policy.
These are sequential in-memory checks, not database concurrency or real
signature validation tests.

### Offline webhook form decoding

`parse_form_body` prepares form-encoded callback input for a future adapter:

```python
from zentomic.webhook import parse_form_body

fields = parse_form_body("Digits=1&Optional=", is_base64_encoded=False)
assert fields == {"Digits": "1", "Optional": ""}
```

Pass the proxy event's `body` and `isBase64Encoded` flag as described in the
[API Gateway proxy contract](https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html).
The helper accepts UTF-8 text or strict base64, with a project limit of 16 KiB
on the decoded body and 128 fields. It also bounds input before decoding.
Empty bodies produce an empty dictionary; blank values and unknown fields are
retained. Names are case-sensitive. Form decoding happens once, including `+`
as space, using [Python's form parser](https://docs.python.org/3/library/urllib.parse.html#urllib.parse.parse_qsl).
Malformed escapes, invalid UTF-8/base64, missing `=`, empty field names, and
duplicate decoded names raise `ValueError`. Error messages omit request data.

This does not expose an endpoint or authenticate a webhook. A future adapter
must enforce POST and `application/x-www-form-urlencoded`, validate the provider
signature against the correct public URL and all form fields, authorize the
workspace, and atomically deduplicate callbacks before routing or persisting
state. Keep the original event for signature validation; do not filter fields,
decode them again, log the body, or treat a successful parse as authentication.
Only after those checks should `fields.get("Digits")` feed `resolve_gather`
with trusted session state. Malformed requests must be rejected, not treated
as silence or allowed to consume a collection attempt.

`parse_form_event` adds offline transport checks around the body decoder:

```python
from zentomic.webhook_event import parse_form_event

fields = parse_form_event({
    "httpMethod": "POST",
    "headers": {"Content-Type": "application/x-www-form-urlencoded"},
    "body": "Digits=1",
    "isBase64Encoded": False,
})
assert fields == {"Digits": "1"}
```

It accepts REST v1 (including an omitted version) and HTTP v2 proxy events,
requires POST, and accepts only form content with an optional UTF-8 charset.
Header names are case-insensitive. Duplicate or conflicting content types are
rejected, including v2 comma-joined duplicates; an exact single-value v1 mirror
in `headers` and `multiValueHeaders` is accepted. These formats follow the
[AWS proxy payload reference](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html).
Other media-type parameters are intentionally unsupported. A missing body is
invalid, while an explicit empty string decodes to `{}`. The encoding flag
defaults to false only when absent. Invalid transport raises `ValueError`
without consuming a collection attempt. The original event is not modified.
This helper does not validate the path, authenticate the sender, or expose a
voice route: signature validation and workspace/session checks remain required
before acting on its result. The existing health handler is unchanged.

### Injected webhook signature gate

`zentomic.authentication.validate_form_event(event, public_url=..., validator=...)`
wraps transport parsing with a required signature-validation dependency. Inject
a bound `twilio.request_validator.RequestValidator(...).validate` method from
trusted application setup, configured with the expected account's auth token.
Twilio recommends its [SDK validator](https://www.twilio.com/docs/usage/security#validating-requests);
this repository does not implement its own signature algorithm or install the SDK.

The gate passes the exact configured HTTPS public URL, every decoded form field,
and the `X-Twilio-Signature` value to the validator. Preserve the public path and
original query string; never derive the URL or expected account from untrusted
Host/forwarded headers or caller fields. Credentials and fragments in the URL
are rejected. The configured URL must also be UTF-8 encodable: lone surrogate
code points fail before the validator is called. Valid Unicode and percent
escapes are passed unchanged, with no normalization or silent repair. This is
not DNS, reachability, or live SDK compatibility validation. Account selection
and secret loading remain adapter responsibilities.

Missing, malformed, duplicate, or conflicting signatures fail closed. Header
names are case-insensitive; an exact v1 single/multivalue mirror is accepted.
Only a validator result of boolean `True` releases the fields. False, truthy
non-booleans, and validator exceptions raise `ValueError` without exposing
dependency error details. Invalid transport never reaches the validator.

Tests inject synthetic validators, not real credentials. They exercise gate
behavior only, not cryptographic correctness or live SDK compatibility. This
does not expose an authenticated endpoint. Production integration still needs
the SDK, trusted configuration, signature integration tests, workspace/session
authorization, and atomic callback deduplication. A valid signature alone does
not prevent replay. Do not route or consume attempts until all checks succeed.

`validate_call_event` adds exact account/call binding on top of that gate:

```python
# example: compile-only
from zentomic.authentication import validate_call_event

# All configuration and session values below must come from trusted setup.
fields = validate_call_event(
    event,
    public_url=configured_callback_url,
    validator=trusted_signature_validator,
    expected_account_sid=session.account_sid,
    expected_call_sid=session.call_sid,
)
```

This integration sketch is not a configured endpoint. The helper first validates
the expected identifiers as nonblank strings, then authenticates the complete
form, and only returns it when `AccountSid` and `CallSid` exactly match the
expected session. Missing, blank, or mismatched fields fail with a generic
`ValueError` that does not include identifiers. No trimming, case conversion,
or provider identifier-format validation is performed. Unknown and blank
optional fields remain available after successful validation.

Select and authorize the session within the intended workspace before supplying
these expected values. Never copy the callback's identifiers into the expected
arguments, which would make the comparison meaningless. Account/call binding
does not establish workspace ownership, verify the callback's current menu
step, or prevent replay. Atomic deduplication and persisted attempt/state checks
are still required before acting on an authenticated callback. The lower-level
`validate_form_event` remains available for non-call form webhooks.

### Confirmed intent routing

`resolve_intent` accepts an intent label from a future classifier and selects
only a target in the configured menu, after explicit caller confirmation:

```python
from zentomic.intent import resolve_intent

target = resolve_intent(
    "technical_support",
    {"sales": "team/Sales", "technical_support": "team/Support"},
    fallback_target="team/Reception",
    confirmed=True,
)
assert target == "team/Support"
```

Confirmation defaults to `False`. Only boolean `True` permits routing; truthy
values such as `"true"` or `1` do not. Unconfirmed, missing, malformed, and unknown
intents use the fallback. Labels match exactly, with no trimming or case
conversion. Menus may be empty; all labels, targets, and the required fallback
must be nonblank UTF-8 encodable strings.
Invalid configuration raises `ValueError`, including
unused routes. Target identifiers are preserved, and the menu is not mutated.
Routing labels use the same encoding requirement as classifier allowlists:
surrogate code points in any configured label are rejected, even without
confirmation or after the confirmation retry budget is exhausted. Valid Unicode
labels retain their exact whitespace and normalization form. Malformed incoming
labels that are absent from a valid menu still fall back; they do not become
configuration errors.

This is an offline policy helper, not speech recognition or an AI integration.
It cannot establish whether a caller confirmed an intent. A future adapter must
derive confirmation from trusted call-session state for that specific intent,
not from model output or an untrusted webhook field. It must also authorize the
menu and all targets within the workspace. Use a human reception target as the
fallback for ambiguous or unconfirmed requests. No calls or messages are sent.

#### Bounded keypad confirmation

`resolve_intent_confirmation` turns one completed confirmation collection into
an immutable `GatherDecision`. Use `render_dtmf_gather` with a trusted prompt
such as "For Support, press 1 to confirm or 2 for reception":

```python
from zentomic.intent import resolve_intent_confirmation

decision = resolve_intent_confirmation(
    "1", "technical_support", {"technical_support": "team/Support"},
    fallback_target="team/Reception", attempts=0,
)
assert (decision.action, decision.target, decision.attempts) == (
    "route", "team/Support", 1,
)
```

Only exact ASCII `1` confirms; `2` immediately returns `fallback`. Silence and
other input return `retry` without a target until the final attempt, then
`fallback`. The default budget is three collections. A valid confirmation on
the last attempt still routes. Missing, malformed, or unknown pending intents
immediately fall back, including for `1`. Each completed collection increments
the count once, unless the budget was already exhausted, in which case it
always falls back without incrementing. All configuration is validated first.

Pass `hangup_digit="9"` to optionally offer an explicit exit during confirmation.
For a known pending intent with budget remaining, that exact digit returns
`GatherDecision("hangup", None, attempts + 1)`, including on the last attempt.
The key must be a single ASCII digit other than the reserved `1` and `2`;
invalid keys are rejected even for unknown intents or exhausted budgets.
The default `None` preserves existing behavior. Missing/unknown intents and
exhausted budgets still fall back, even if the exit key was pressed. Mention
the configured exit in the prompt, persist a `hangup` decision as terminal,
and render `render_hangup()` without forwarding or collecting another digit.

This is an offline policy, not proof that a real caller confirmed. A future
adapter must authenticate the callback, bind it to the current confirmation
step and its trusted pending intent, atomically deduplicate it, and persist the
returned count and decision before rendering a retry or forwarding. Never use
caller/model-supplied intent, confirmation flags, or attempt counts as that
state. Do not accept a late confirmation for a changed intent or completed
step. Route targets and the human fallback require workspace authorization.
No new endpoint, session store, AI request, or telephony operation is added.

## Local development

Use Python 3.12 or newer. No package installation, accounts, or secrets required.
From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m unittest -v
python -m compileall -q zentomic tests
python -m zentomic
```

On Windows PowerShell, create the environment with `py -3 -m venv .venv`
and activate it with `.venv\Scripts\Activate.ps1`. The remaining `python`
commands are the same. The last command invokes a synthetic health request
locally and prints the proxy response; it does not start a server or use the
network.

The test directory is an importable package, so default discovery from the
repository root (`python -m unittest` or `python -m unittest discover`) runs
the full offline suite. Explicit discovery with `python -m unittest discover
-s tests -v` remains supported. A discovery regression test checks that every
`test_*.py` module is included, preventing a misleading empty test run.

The suite also runs each README Python example in a separate namespace with
an empty environment and socket creation blocked, so examples include their
own imports. The adapter sketch marked `# example: compile-only` is checked
for syntax but not executed because it requires trusted configuration and
session state. These checks are not a sandbox for untrusted documentation and
do not validate live integrations. Run just these checks with
`python -m unittest tests.test_readme -v`.

The future Lambda entry point is `zentomic.handler.lambda_handler`.

### Optional local package installation

`pyproject.toml` makes the scaffold installable so its helpers and
`python -m zentomic` can be used outside the repository directory. Runtime
dependencies remain empty; the root-based test and smoke commands above still
require no installation. Packaging uses
[setuptools configuration](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html).

For editable development in an activated virtual environment, run
`python -m pip install -e .` from the repository root. This may download build
tools, but does not contact AWS, Twilio, or OpenAI. For an offline wheel build,
use an environment with pip, setuptools 68 or newer, and wheel already installed:

```sh
python -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir dist .
python -m pip install --no-index --no-deps dist/zentomic-0.1.0-py3-none-any.whl
```

The wheel includes only the `zentomic` package and distribution metadata, not
the test package or local configuration files. Keep running the test suite
from the repository root. Building or installing locally does not publish a
package or produce a deployable Lambda bundle; there is no release or
deployment automation. Version `0.1.0` identifies this offline scaffold, not
the production service.

### Cross-module tests

`tests/test_voice_flow.py` exercises the offline cross-module contract: a
synthetic authenticated callback, bounded confirmation, forwarding XML, and a
terminal outcome, including the one-fallback limit. It covers both proxy
formats and body encodings, rejected account/call bindings, and preservation
of test-owned attempt/fallback state despite conflicting signed form fields.
The test harness is not an application adapter: its fake validator does not
verify cryptography, and its local state does not implement persistence,
workspace authorization, current-step binding, or replay prevention.

`tests/test_speech_flow.py` extends that contract through speech admission,
a mocked classifier, strict response parsing, and a separately authenticated
confirmation before forwarding XML. It checks that rejected callbacks, silence,
oversized speech, and exhausted budgets never invoke the classifier; malformed
model output cannot become a confirmed route; and signed extra fields cannot
supply confirmation or reset test-owned budgets. These tests use synthetic
transcripts and share the same fake-authentication and persistence limitations.

`tests/test_budget_flow.py` demonstrates the deadline boundary around that
mocked classification step: authenticate first, read a fresh clock, pass the
remaining-budget timeout to the dependency, and read the clock again before
admitting its result. It checks insufficient budget, late-result rejection,
cleanup reserves, timeout granularity, and one unchanged deadline across
callbacks. Signed form fields cannot supply test-owned timing configuration.
The example also samples a trusted mock runtime's remaining invocation duration
after authentication on every collection. Integration checks cover the shorter
invocation cap, cleanup reserve and rounding, refusing to start when too little
runtime remains, ignoring caller-supplied runtime values, and a new invocation
never extending the fixed call deadline. Before parsing a classifier result,
the example samples both budgets again and discards output if either has
expired, even when the other still has time. Tests script runtime exhaustion
during classification, the positive one-millisecond boundary, and invalid
runtime readings. This final gate only checks expiry: it does not reserve time
for parsing, persistence, or transport. Any subsequent operation still needs its
own fresh admission check with an appropriate minimum and cleanup reserve.
Rejected callbacks read neither clock nor runtime budget. The runtime mock
only governs admission; scripted readings do not enforce cancellation or model
an actual Lambda timeout. An exhausted runtime may prevent even a serialized
hangup from reaching the provider.
The mock's `timeout_ms` keyword is not an SDK API, and these tests do not prove
provider cancellation, enforce a live timeout, or implement a production
adapter. A usable pending label still requires separate caller confirmation.

## Layout

- `zentomic/handler.py`: health route and API Gateway proxy response handling.
- `zentomic/routing.py`: deterministic single-digit menu selection and fallback.
- `zentomic/gather.py`: bounded collection retries with immutable decisions.
- `zentomic/budget.py`: fixed whole-call deadline admission with injected timestamps.
- `zentomic/speech.py`: bounded speech result admission before classification.
- `zentomic/classification.py`: strict JSON admission of allowlisted pending intents.
- `zentomic/dial.py`: one-fallback Number dial outcome policy.
- `zentomic/call_status.py`: strict terminal versus active call-leg classification.
- `zentomic/intent.py`: allowlisted intent routing gated on caller confirmation.
- `zentomic/twiml.py`: offline collection, forwarding, and terminal response rendering.
- `zentomic/webhook.py`: bounded form decoding with duplicate-field rejection.
- `zentomic/webhook_event.py`: offline POST/form proxy transport validation.
- `zentomic/authentication.py`: injected signature gate and trusted call-session binding.
- `zentomic/__main__.py`: credential-free local smoke check.
- `tests/test_handler.py`: offline standard-library unit tests.
- `tests/test_readme.py`: isolated executable Python examples and sketch syntax.
- `tests/test_routing.py`: menu validation, fallback, and side-effect tests.
- `tests/test_gather.py`: retry budgets, exhaustion, and input validation tests.
- `tests/test_budget.py`: shared deadlines, exact boundaries, and offline validation.
- `tests/test_budget_flow.py`: authenticated classifier timeout and late-result contracts.
- `tests/test_intent.py`: confirmation, exact matching, and intent safety tests.
- `tests/test_intent_confirmation.py`: bounded keypad confirmation and renderer composition.
- `tests/test_twiml.py`: collection/ending structure, XML escaping, and renderer limits.
- `tests/test_speech_gather.py`: speech-only XML, timeout bounds, and offline safety.
- `tests/test_speech.py`: speech budgets, text limits, and confirmation separation.
- `tests/test_classification.py`: classifier schema, byte limits, and confirmation separation.
- `tests/test_speech_flow.py`: authenticated speech-to-confirmation cross-module contracts.
- `tests/test_dial.py`: forwarding structure, destination validation, and offline safety.
- `tests/test_dial_result.py`: outcome policy, fallback exhaustion, and offline composition.
- `tests/test_call_status.py`: lifecycle values, rejection, and authenticated composition.
- `tests/test_call_status_flow.py`: authenticated lifecycle sequences and rejected updates.
- `tests/test_webhook.py`: encoding, parser limits, and offline routing integration.
- `tests/test_webhook_event.py`: proxy formats, media types, and transport rejection.
- `tests/test_authentication.py`: signature gate, dependency failures, and privacy.
- `tests/test_call_authentication.py`: account/call binding and rejection before routing.

## Development boundaries

Keep this public repository free of credentials, account identifiers, real phone
numbers, call transcripts, and customer data. Use synthetic fixtures only.
Local `.env` files are ignored and are not loaded by the scaffold.

Future increments can add mocked service adapters, call-session state, and
webhook validation. The signature gate has no configured production validator;
end-to-end authentication, workspace isolation, cryptographic signature
verification, persistence, and deployment are not implemented. Do not connect
this scaffold to real call traffic until those boundaries are designed and
tested. Do not add automatic deployments or paid service calls as part of
routine development.
