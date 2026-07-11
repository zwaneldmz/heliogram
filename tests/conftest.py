"""Shared test helpers.

assert_import_stays_torch_free: the repo-wide "CPU-only import boundary" invariant, checked in
a FRESH SUBPROCESS. Several test files used to assert `"torch" not in sys.modules` in-process;
that was order-dependent the moment any OTHER collected test file legitimately imported torch
(the GPU-path contract tests do, whenever torch is installed) -- pytest imports every collected
module before running the first test, so the in-process assertion failed for reasons that had
nothing to do with the module under test. A subprocess sees only the import graph of the module
it was asked to import, which is the actual invariant.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def assert_import_stays_torch_free(*module_names: str) -> None:
    """Import each module in a fresh subprocess and assert torch/transformers were not pulled
    in as a side effect. Raises AssertionError with the subprocess's stderr on failure."""
    imports = "; ".join(f"import {name}" for name in module_names)
    code = (
        "import sys; "
        f"{imports}; "
        "assert 'torch' not in sys.modules, 'torch was imported'; "
        "assert 'transformers' not in sys.modules, 'transformers was imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"importing {module_names} pulled in torch/transformers (or failed outright):\n"
        f"{result.stderr}"
    )
