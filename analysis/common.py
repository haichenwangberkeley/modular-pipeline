from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: Path | str) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(payload: Any, path: Path | str) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return output_path


def write_text(content: str, path: Path | str) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(content)
    return output_path


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def stable_hash(payload: Any) -> str:
    return sha256_text(stable_json_dumps(payload))


def list_root_files(path: Path | str) -> list[Path]:
    base = Path(path)
    if not base.exists():
        return []
    return sorted(
        candidate
        for candidate in base.iterdir()
        if candidate.suffix == ".root" and not candidate.name.startswith("._")
    )


def flatten(items: Iterable[Iterable[Any]]) -> list[Any]:
    out: list[Any] = []
    for group in items:
        out.extend(group)
    return out


def finite_or_default(value: float | None, default: float = 1.0) -> float:
    if value is None:
        return default
    if not math.isfinite(value):
        return default
    return float(value)
