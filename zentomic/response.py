"""Offline API Gateway proxy envelopes for trusted, rendered TwiML."""

from typing import Any


MAX_TWIML_BYTES = 64 * 1024


def twiml_response(xml: str) -> dict[str, Any]:
    """Wrap application-generated TwiML in an explicit v1/v2 success response.

    Supply output from zentomic.twiml renderers, never caller/model-supplied XML.
    This checks text encoding and applies a local 64 KiB body limit, not XML
    syntax, TwiML verbs, authorization, or safe destinations. It does not
    authenticate, persist state, send a response, or expose a voice endpoint.
    Authentication failures need their own non-success response and must not
    use this success-only helper.
    """
    error = (
        "xml must be a nonblank UTF-8 encodable string "
        "of at most 65536 bytes"
    )
    # UTF-8 uses at least one byte per character. Reject clearly oversized
    # output before scanning or encoding it, then enforce the encoded boundary.
    if (not isinstance(xml, str) or len(xml) > MAX_TWIML_BYTES
            or not xml.strip()):
        raise ValueError(error)
    try:
        encoded = xml.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(error) from None
    if len(encoded) > MAX_TWIML_BYTES:
        raise ValueError(error)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/xml; charset=utf-8",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
        "body": xml,
        "isBase64Encoded": False,
    }
