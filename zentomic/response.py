"""Offline API Gateway proxy envelopes for trusted, rendered TwiML."""

from typing import Any


def twiml_response(xml: str) -> dict[str, Any]:
    """Wrap application-generated TwiML in an explicit v1/v2 success response.

    Supply output from zentomic.twiml renderers, never caller/model-supplied XML.
    This checks text encoding only, not XML syntax, TwiML verbs, authorization,
    or safe destinations. It does not authenticate, persist state, send a
    response, or expose a voice endpoint. Authentication failures need their
    own non-success response and must not use this success-only helper.
    """
    error = "xml must be a nonblank UTF-8 encodable string"
    if not isinstance(xml, str) or not xml.strip():
        raise ValueError(error)
    try:
        xml.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(error) from None
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/xml; charset=utf-8",
            "Cache-Control": "no-store",
        },
        "body": xml,
        "isBase64Encoded": False,
    }
