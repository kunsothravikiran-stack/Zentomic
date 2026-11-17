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
    # Renderers return exact built-in strings. Reject subclasses before any
    # length, whitespace, or encoding operation can invoke overridden code.
    if (type(xml) is not str or len(xml) > MAX_TWIML_BYTES
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
            "Content-Length": str(len(encoded)),
            "Cache-Control": "no-store",
            # Twilio ignores browser policies, but these keep a TwiML endpoint
            # inert if its URL is accidentally opened or embedded in a browser.
            "Content-Security-Policy": (
                "default-src 'none'; frame-ancestors 'none'; sandbox"
            ),
            # Require explicitly same-origin or opted-in subresources if an
            # XML response is ever opened as a browser document.
            "Cross-Origin-Embedder-Policy": "require-corp",
            # Isolate accidental top-level navigation from its opener so the
            # XML response cannot retain a cross-origin window relationship.
            "Cross-Origin-Opener-Policy": "same-origin",
            # Prevent browsers from embedding the XML as a cross-origin
            # subresource and avoid leaking its callback URL in Referer.
            "Cross-Origin-Resource-Policy": "same-origin",
            # A TwiML callback is machine-readable XML and never needs access
            # to browser-controlled device capabilities if opened directly.
            "Permissions-Policy": (
                "camera=(), geolocation=(), microphone=(), payment=(), usb=()"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            # Keep public callback URLs out of search results if discovered.
            "X-Robots-Tag": "noindex, nofollow",
        },
        "body": xml,
        "isBase64Encoded": False,
    }
