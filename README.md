# Zentomic

Modern business IVR and AI call routing, intended for Python AWS Lambda,
API Gateway, DynamoDB, Twilio, and OpenAI.

## Current scope

This repository starts with an offline, dependency-free Python scaffold. It is
not an export of an existing production service. There are no cloud resources,
deployment workflows, credentials, telephony operations, or AI API calls here.

Implemented: a Lambda-compatible `GET /health` liveness endpoint supporting
API Gateway REST API (v1) and HTTP API (v2) proxy events. It returns
`{"status":"ok","service":"zentomic"}` without checking external services.
Other paths return 404; other methods on `/health` return 405 with `Allow: GET`.
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

## Local development

Use Python 3.12 or newer. No package installation, accounts, or secrets required.
From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m compileall -q zentomic tests
python -m zentomic
```

On Windows PowerShell, create the environment with `py -3 -m venv .venv`
and activate it with `.venv\Scripts\Activate.ps1`. The remaining `python`
commands are the same. The last command invokes a synthetic health request
locally and prints the proxy response; it does not start a server or use the
network.

The future Lambda entry point is `zentomic.handler.lambda_handler`.

## Layout

- `zentomic/handler.py`: health route and API Gateway proxy response handling.
- `zentomic/routing.py`: deterministic single-digit menu selection and fallback.
- `zentomic/gather.py`: bounded collection retries with immutable decisions.
- `zentomic/intent.py`: allowlisted intent routing gated on caller confirmation.
- `zentomic/twiml.py`: offline, XML-safe single-digit collection rendering.
- `zentomic/webhook.py`: bounded form decoding with duplicate-field rejection.
- `zentomic/webhook_event.py`: offline POST/form proxy transport validation.
- `zentomic/__main__.py`: credential-free local smoke check.
- `tests/test_handler.py`: offline standard-library unit tests.
- `tests/test_routing.py`: menu validation, fallback, and side-effect tests.
- `tests/test_gather.py`: retry budgets, exhaustion, and input validation tests.
- `tests/test_intent.py`: confirmation, exact matching, and intent safety tests.
- `tests/test_twiml.py`: collection attributes, XML escaping, and renderer limits.
- `tests/test_webhook.py`: encoding, parser limits, and offline routing integration.
- `tests/test_webhook_event.py`: proxy formats, media types, and transport rejection.

## Development boundaries

Keep this public repository free of credentials, account identifiers, real phone
numbers, call transcripts, and customer data. Use synthetic fixtures only.
Local `.env` files are ignored and are not loaded by the scaffold.

Future increments can add mocked service adapters, call-session state, and
webhook validation. Authentication, workspace isolation, Twilio signature
verification, persistence, and deployment are not implemented. Do not connect
this scaffold to real call traffic until those boundaries are designed and
tested. Do not add automatic deployments or paid service calls as part of
routine development.
