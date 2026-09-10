"""Offline TwiML serialization for collection, forwarding, and call endings."""

import re
from xml.etree.ElementTree import Element, SubElement, tostring


def _validate_prompt(prompt: str) -> None:
    """Apply shared text limits before serializing a Say element."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
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
    return tostring(response, encoding="unicode")


def render_dial(
    numbers: list[str] | tuple[str, ...], *, action_path: str, timeout: int = 20,
) -> str:
    """Serialize simultaneous forwarding to 1-10 trusted phone destinations.

    Numbers must already be resolved and authorized within the call workspace.
    This checks E.164-style syntax only, not assignment, ownership, or safety.
    No caller/model-supplied destinations, SDK calls, or endpoint are provided.
    timeout bounds ringing, not conversation length or provider billing.
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

    response = Element("Response")
    dial = SubElement(response, "Dial", {
        "action": action_path, "method": "POST", "timeout": str(timeout),
        "sequential": "false", "record": "do-not-record",
    })
    for number in destinations:
        SubElement(dial, "Number").text = number
    return tostring(response, encoding="unicode")


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
    return tostring(response, encoding="unicode")
