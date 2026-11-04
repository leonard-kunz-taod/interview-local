"""The one rule for keeping Hugging Face tokens out of relayed and reported text."""

import re

REDACTED = "[Token redacted]"
HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]+")


def redacted(text: str, token: str | None) -> str:
    """Text with the exact token and anything shaped like a Hub token replaced.

    Args:
        text: Output or an error message that may contain a token.
        token: The token in use; None or empty when there is none.

    Returns:
        The text with every token occurrence replaced by REDACTED.
    """
    if token:
        text = text.replace(token, REDACTED)
    return HF_TOKEN_PATTERN.sub(REDACTED, text)
