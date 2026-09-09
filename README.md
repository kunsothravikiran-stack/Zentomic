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
- `zentomic/__main__.py`: credential-free local smoke check.
- `tests/test_handler.py`: offline standard-library unit tests.
- `tests/test_routing.py`: menu validation, fallback, and side-effect tests.

## Development boundaries

Keep this public repository free of credentials, account identifiers, real phone
numbers, call transcripts, and customer data. Use synthetic fixtures only.
Local `.env` files are ignored and are not loaded by the scaffold.

Future increments can add intent routing, mocked service adapters, and
webhook validation. Authentication, workspace isolation, Twilio signature
verification, persistence, and deployment are not implemented. Do not connect
this scaffold to real call traffic until those boundaries are designed and
tested. Do not add automatic deployments or paid service calls as part of
routine development.
