"""Offline TwiML for collection, forwarding, transitions, and call endings."""

import re
from xml.etree.ElementTree import Element, SubElement, tostring


def _serialize(response: Element) -> str:
    """Preserve prompt carriage returns through XML end-of-line handling."""
    # Escape markup first so literal caller-like text such as "&#13;" stays
    # text. Raw CR/CRLF would otherwise be normalized to LF by XML parsers.
    return tostring(response, encoding="unicode").replace("\r", "&#13;")


def _validate_prompt(prompt: str) -> None:
    """Apply shared text limits before serializing a Say element."""
    # Bound work before stripping: oversized padded text could otherwise
    # require a full scan and an unnecessary copy before being rejected.
    if not isinstance(prompt, str) or len(prompt) > 1000 or not prompt.strip():
        raise ValueError("prompt must be a nonblank string of at most 1000 characters")
    # ElementTree escapes markup, but does not reject XML 1.0-invalid characters.
    if any(
        not (char in "\t\n\r" or 0x20 <= ord(char) <= 0xD7FF
             or 0xE000 <= ord(char) <= 0xFFFD or 0x10000 <= ord(char) <= 0x10FFFF)
        for char in prompt
    ):
        raise ValueError("prompt must contain only valid XML characters")


def _validate_action_path(action_path: str) -> None:
    if not isinstance(action_path, str) or not re.fullmatch(
        r"/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*/?", action_path
    ):
        raise ValueError("action_path must be a root-relative path of named segments")


def render_dtmf_gather(prompt: str, *, action_path: str, timeout: int = 5) -> str:
    """Render one collection, including a callback when no digit is received.

    This intentionally supports only root-relative application callback paths,
    not external URLs or inline TwiML supplied through Twilio's Calls API.
    Use trusted application configuration for the prompt and callback path.
    Rendering does not send TwiML, authenticate callbacks, or manage retries.
    """
    _validate_prompt(prompt)
    _validate_action_path(action_path)
    if type(timeout) is not int or not 1 <= timeout <= 60:
        raise ValueError("timeout must be an integer from 1 to 60 seconds")

    response = Element("Response")
    gather = SubElement(response, "Gather", {
        "input": "dtmf",
        "numDigits": "1",
        "finishOnKey": "",
        "action": action_path,
        "method": "POST",
        "actionOnEmptyResult": "true",
        "timeout": str(timeout),
    })
    SubElement(gather, "Say").text = prompt
    return _serialize(response)


def render_speech_gather(
    prompt: str, *, action_path: str, timeout: int = 5, speech_timeout: int = 2,
) -> str:
    """Serialize an English speech-only collection for a future intent adapter.

    timeout bounds waiting for input; speech_timeout is the pause that ends an
    utterance, not a maximum recording duration or whole-call budget. Both are
    project-limited to 1-60 seconds. This does not transcribe, classify, confirm,
    retry, authenticate callbacks, or contact a speech provider.
    """
    _validate_prompt(prompt)
    _validate_action_path(action_path)
    for name, value in (("timeout", timeout), ("speech_timeout", speech_timeout)):
        if type(value) is not int or not 1 <= value <= 60:
            raise ValueError(f"{name} must be an integer from 1 to 60 seconds")

    response = Element("Response")
    gather = SubElement(response, "Gather", {
        "input": "speech", "language": "en-US",
        "action": action_path, "method": "POST", "actionOnEmptyResult": "true",
        "timeout": str(timeout), "speechTimeout": str(speech_timeout),
    })
    SubElement(gather, "Say").text = prompt
    return _serialize(response)


def render_dial(
    numbers: list[str] | tuple[str, ...], *, action_path: str, timeout: int = 20,
    time_limit: int = 14400,
) -> str:
    """Serialize simultaneous forwarding to 1-10 trusted phone destinations.

    Numbers must already be resolved and authorized within the call workspace.
    This checks E.164-style syntax only, not assignment, ownership, or safety.
    No caller/model-supplied destinations, SDK calls, or endpoint are provided.
    timeout bounds ringing; time_limit bounds this Dial's connected duration.
    Neither setting is a whole-session budget or a provider billing cap.
    """
    if not isinstance(numbers, (list, tuple)) or not 1 <= len(numbers) <= 10:
        raise ValueError("numbers must be a list or tuple of 1 to 10 destinations")
    destinations = tuple(numbers)
    for number in destinations:
        if not isinstance(number, str) or not re.fullmatch(r"\+[1-9][0-9]{1,14}", number):
            raise ValueError("destinations must use E.164-style phone number syntax")
    if len(set(destinations)) != len(destinations):
        raise ValueError("destinations must be unique")
    _validate_action_path(action_path)
    if type(timeout) is not int or not 5 <= timeout <= 60:
        raise ValueError("timeout must be an integer from 5 to 60 seconds")
    if type(time_limit) is not int or not 1 <= time_limit <= 14400:
        raise ValueError("time_limit must be an integer from 1 to 14400 seconds")

    response = Element("Response")
    dial = SubElement(response, "Dial", {
        "action": action_path, "method": "POST", "timeout": str(timeout),
        "timeLimit": str(time_limit),
        "sequential": "false", "record": "do-not-record",
    })
    for number in destinations:
        SubElement(dial, "Number").text = number
    return _serialize(response)


def render_redirect(*, action_path: str) -> str:
    """Serialize a POST transition to a trusted, hosted application voice step.

    Reuse the root-relative callback path policy; caller/model URLs are never
    appropriate. Persist the authorized next step before returning this XML.
    The receiving adapter must authenticate again and enforce the existing
    session deadline and transition budget, not reset them on redirection.
    This neither performs a request nor prevents redirect loops, and is not
    intended for inline TwiML without a hosted base URL.
    """
    _validate_action_path(action_path)
    response = Element("Response")
    SubElement(response, "Redirect", {"method": "POST"}).text = action_path
    return _serialize(response)


def render_hangup(prompt: str | None = None) -> str:
    """Render an explicit terminal response with an optional trusted farewell.

    None omits Say; supplied text uses the same limits as the menu prompt.
    This only serializes XML, not a live call operation or a routing decision.
    Hangup is not Reject and does not prevent answering or provider billing.
    """
    if prompt is not None:
        _validate_prompt(prompt)
    response = Element("Response")
    if prompt is not None:
        SubElement(response, "Say").text = prompt
    SubElement(response, "Hangup")
    return _serialize(response)
