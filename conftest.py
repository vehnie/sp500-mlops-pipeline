"""Repository-wide pytest configuration."""

from __future__ import annotations

import getpass
import os
import re
import subprocess

import pytest


def _current_identity() -> str:
    """Return the account from the active Windows security token."""
    if os.name == "nt":
        try:
            return subprocess.check_output(
                ["whoami"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass

    return getpass.getuser()


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest temp files local and isolated by Windows user."""
    if config.option.basetemp is not None:
        return

    identity = re.sub(r"[^A-Za-z0-9_.-]", "_", _current_identity())
    config.option.basetemp = config.rootpath / ".pytest_cache" / f"tmp-{identity}"
