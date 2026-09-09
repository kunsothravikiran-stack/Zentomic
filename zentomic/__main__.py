"""Invoke a synthetic health request locally, without network access."""

import json

from zentomic.handler import lambda_handler


if __name__ == "__main__":
    event = {
        "version": "2.0",
        "rawPath": "/health",
        "requestContext": {"http": {"method": "GET"}},
    }
    print(json.dumps(lambda_handler(event, None), indent=2))
