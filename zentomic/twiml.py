"""Offline TwiML serialization for a hosted, single-digit IVR menu."""

import re
from xml.etree.ElementTree import Element, SubElement, tostring


def render_dtmf_gather(prompt: str, *, action_path: str, timeout: int = 5) -> str:
    """Render one collection, including a callback when no digit is received.

    This intentionally supports only root-relative application callback paths,
    not external URLs or inline TwiML supplied through Twilio's Calls API.
    Use trusted application configuration for the prompt and callback path.
    Rendering does not send TwiML, authenticate callbacks, or manage retries.
    """
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
        raise ValueError("prompt must be a nonblank string of at most 1000 characters")
    # ElementTree escapes markup, but does not reject XML 1.0-invalid characters.
    if any(
        not (char in "\t\n\r" or 0x20 <= ord(char) <= 0xD7FF
             or 0xE000 <= ord(char) <= 0xFFFD or 0x10000 <= ord(char) <= 0x10FFFF)
        for char in prompt
    ):
        raise ValueError("prompt must contain only valid XML characters")
    if not isinstance(action_path, str) or not re.fullmatch(
        r"/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*/?", action_path
    ):
        raise ValueError("action_path must be a root-relative path of named segments")
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
