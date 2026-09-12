"""Invoke the health handler with a synthetic, stdin, or file event locally."""

import argparse
import json
import math
import sys

from zentomic.handler import lambda_handler


MAX_EVENT_BYTES = 64 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-JSON constant")


def _finite_float(value: str) -> float:
    # parse_constant rejects literal Infinity, but not JSON numbers like 1e400
    # that overflow Python's float during decoding, including nested values.
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds finite float range")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--stdin", action="store_true",
                        help="read one UTF-8 JSON proxy event from stdin (at most 64 KiB)")
    source.add_argument("--event-file", metavar="PATH",
                        help="read one local UTF-8 JSON proxy event file (at most 64 KiB)")
    args = parser.parse_args(argv)
    event = {
        "version": "2.0",
        "rawPath": "/health",
        "requestContext": {"http": {"method": "GET"}},
    }
    if args.stdin or args.event_file is not None:
        try:
            # Read bytes, not text characters, to bound multibyte input too.
            if args.event_file is not None:
                with open(args.event_file, "rb") as stream:
                    raw = stream.read(MAX_EVENT_BYTES + 1)
            else:
                raw = sys.stdin.buffer.read(MAX_EVENT_BYTES + 1)
            if len(raw) > MAX_EVENT_BYTES:
                raise ValueError("event too large")
            # Accept an optional leading UTF-8 BOM from local editor exports.
            # Keep decoding explicit: json.loads(bytes) also accepts UTF-16/32.
            event = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                               parse_constant=_reject_constant, parse_float=_finite_float)
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
        except (ValueError, RecursionError, OSError):
            print("Invalid event: provide one UTF-8 JSON object of at most 65536 bytes.",
                  file=sys.stderr)
            return 2
    print(json.dumps(lambda_handler(event, None), indent=2))
    # A handler 4xx is a successfully replayed request, not a CLI input error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
