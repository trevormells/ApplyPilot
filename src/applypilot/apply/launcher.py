"""Deprecated compatibility wrapper for the browser-use launcher.

`applypilot.apply.launcherv2` is the canonical apply runtime. This module
remains only to preserve older imports while steering callers to the new path.
"""

from __future__ import annotations

import warnings

from applypilot.apply.launcherv2 import *  # noqa: F401,F403

warnings.warn(
    "applypilot.apply.launcher is deprecated; import applypilot.apply.launcherv2 instead.",
    DeprecationWarning,
    stacklevel=2,
)
