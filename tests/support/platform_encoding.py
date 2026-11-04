"""Emulate a platform whose default text encoding is not UTF-8."""

import contextlib
import io
from unittest.mock import patch

WINDOWS_DEFAULT_ENCODING = "cp1252"


@contextlib.contextmanager
def windows_default_encoding():
    """Text files opened without an encoding use cp1252, as on Windows."""
    resolve = io.text_encoding

    def platform_default(encoding, stacklevel=2):
        if encoding is None:
            return WINDOWS_DEFAULT_ENCODING
        return resolve(encoding, stacklevel)

    with patch("io.text_encoding", platform_default):
        yield
