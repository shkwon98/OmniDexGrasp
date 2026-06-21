"""Compatibility shims for legacy third-party packages."""
from __future__ import annotations

import inspect
import warnings

import numpy as np


def install_chumpy_compat() -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name, value in aliases.items():
            if not hasattr(np, name):
                setattr(np, name, value)

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
