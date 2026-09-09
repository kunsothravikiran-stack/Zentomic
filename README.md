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
- `zentomic/__main__.py`: credential-free local smoke check.
- `tests/test_handler.py`: offline standard-library unit tests.

## Development boundaries

Keep this public repository free of credentials, account identifiers, real phone
numbers, call transcripts, and customer data. Use synthetic fixtures only.
Local `.env` files are ignored and are not loaded by the scaffold.

Future increments can add pure routing logic, mocked service adapters, and
webhook validation. Authentication, workspace isolation, Twilio signature
verification, persistence, and deployment are not implemented. Do not connect
this scaffold to real call traffic until those boundaries are designed and
tested. Do not add automatic deployments or paid service calls as part of
routine development.
