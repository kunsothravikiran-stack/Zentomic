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
- `zentomic/__main__.py`: credential-free local smoke check.
- `tests/test_handler.py`: offline standard-library unit tests.
- `tests/test_routing.py`: menu validation, fallback, and side-effect tests.
- `tests/test_gather.py`: retry budgets, exhaustion, and input validation tests.
- `tests/test_intent.py`: confirmation, exact matching, and intent safety tests.

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
