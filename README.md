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
the required fallback must be nonblank strings; invalid configuration raises
`ValueError` before selecting a route. Input is not trimmed or coerced, so
multi-digit values, whitespace, Unicode digits, `*`, and `#` use the fallback.
The resolver does not log input, mutate the menu, or access the environment or
network. Identifiers are returned unchanged. A future integration must load an
authorized workspace's menu and resolve its targets within that same workspace;
this helper does not perform authentication or workspace authorization.

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
are rejected. Account selection and secret loading remain adapter responsibilities.

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
must be nonblank strings. Invalid configuration raises `ValueError`, including
unused routes. Target identifiers are preserved, and the menu is not mutated.

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

The future Lambda entry point is `zentomic.handler.lambda_handler`.

`tests/test_voice_flow.py` exercises the offline cross-module contract: a
synthetic authenticated callback, bounded confirmation, forwarding XML, and a
terminal outcome, including the one-fallback limit. It covers both proxy
formats and body encodings, rejected account/call bindings, and preservation
of test-owned attempt/fallback state despite conflicting signed form fields.
The test harness is not an application adapter: its fake validator does not
verify cryptography, and its local state does not implement persistence,
workspace authorization, current-step binding, or replay prevention.

## Layout

- `zentomic/handler.py`: health route and API Gateway proxy response handling.
- `zentomic/routing.py`: deterministic single-digit menu selection and fallback.
- `zentomic/gather.py`: bounded collection retries with immutable decisions.
- `zentomic/dial.py`: one-fallback Number dial outcome policy.
- `zentomic/intent.py`: allowlisted intent routing gated on caller confirmation.
- `zentomic/twiml.py`: offline collection, forwarding, and terminal response rendering.
- `zentomic/webhook.py`: bounded form decoding with duplicate-field rejection.
- `zentomic/webhook_event.py`: offline POST/form proxy transport validation.
- `zentomic/authentication.py`: injected signature gate and trusted call-session binding.
- `zentomic/__main__.py`: credential-free local smoke check.
- `tests/test_handler.py`: offline standard-library unit tests.
- `tests/test_routing.py`: menu validation, fallback, and side-effect tests.
- `tests/test_gather.py`: retry budgets, exhaustion, and input validation tests.
- `tests/test_intent.py`: confirmation, exact matching, and intent safety tests.
- `tests/test_intent_confirmation.py`: bounded keypad confirmation and renderer composition.
- `tests/test_twiml.py`: collection/ending structure, XML escaping, and renderer limits.
- `tests/test_speech_gather.py`: speech-only XML, timeout bounds, and offline safety.
- `tests/test_dial.py`: forwarding structure, destination validation, and offline safety.
- `tests/test_dial_result.py`: outcome policy, fallback exhaustion, and offline composition.
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
