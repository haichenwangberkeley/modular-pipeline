from __future__ import annotations

import os
import subprocess
from pathlib import Path

from analysis.common import ensure_dir, utcnow_iso, write_json


WORKSPACE = Path(__file__).resolve().parents[1]
ROOTENV_PYTHON = WORKSPACE / ".rootenv" / "bin" / "python"
ROOTENV_EXISTS = ROOTENV_PYTHON.exists()
ACTIVE_PYTHON = Path(os.sys.executable)


def runtime_python() -> Path:
    return ROOTENV_PYTHON if ROOTENV_EXISTS else ACTIVE_PYTHON


def runtime_context() -> dict:
    return {
        "workspace": str(WORKSPACE),
        "python": str(runtime_python()),
        "rootenv_exists": ROOTENV_EXISTS,
        "timestamp_utc": utcnow_iso(),
    }


def check_pyroot() -> dict:
    try:
        version = subprocess.check_output(
            [str(runtime_python()), "-c", "import ROOT; print(ROOT.gROOT.GetVersion())"],
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - runtime dependent
        reason = str(exc)
        if not ROOTENV_EXISTS:
            reason = f"workspace_rootenv_missing; active_python_check_failed: {reason}"
        return {"available": False, "reason": reason}
    return {"available": True, "version": version, "python": str(runtime_python())}


def write_runtime_recovery(path: Path | str) -> Path:
    pyroot = check_pyroot()
    payload = {
        "status": "ok" if pyroot.get("available") else "failed",
        "workspace": str(WORKSPACE),
        "rootenv_python": str(ROOTENV_PYTHON),
        "active_python": str(ACTIVE_PYTHON),
        "pyroot": pyroot,
        "timestamp_utc": utcnow_iso(),
        "notes": [
            "Prefer a repo-local .rootenv for reproducible RooFit stages, but fall back to the active Python interpreter when it already provides PyROOT."
        ],
    }
    return write_json(payload, path)
