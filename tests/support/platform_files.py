"""Emulate a case-sensitive file system on a case-insensitive one."""

import contextlib
import errno
import os
from pathlib import Path
from unittest.mock import patch


@contextlib.contextmanager
def case_sensitive_file_names():
    """A file name matches only in its exact case, as on Linux."""
    open_file, is_file = Path.open, Path.is_file

    def exact(path):
        return path.parent.is_dir() and path.name in os.listdir(path.parent)

    def opened(self, mode="r", *args, **kwargs):
        if "r" in mode and not exact(self):
            raise FileNotFoundError(errno.ENOENT, "No such file", str(self))
        return open_file(self, mode, *args, **kwargs)

    def found(self):
        return is_file(self) and exact(self)

    with patch.object(Path, "open", opened), patch.object(Path, "is_file", found):
        yield
