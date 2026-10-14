"""Tamper-evident approval log.

Every approval or rejection is one line of JSON. Each line contains the hash of the line before it, so
changing, deleting or reordering an old entry breaks every hash after it, and `verify()` says so.
This does not stop someone with file access from rewriting the whole file; it makes quiet edits to
the middle of the history detectable, which is the point of an audit trail.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

GENESIS = "0" * 64


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _lines(self) -> list[str]:
        if not self.path.is_file():
            return []
        return [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def append(self, entry: dict) -> dict:
        with self._lock:
            lines = self._lines()
            prev = json.loads(lines[-1])["hash"] if lines else GENESIS
            body = dict(entry, prev_hash=prev)
            body["hash"] = sha256_text(canonical(body))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(canonical(body) + "\n")
            return body

    def count(self) -> int:
        with self._lock:
            return len(self._lines())

    def entries(self, limit: int = 50) -> list[dict]:
        out = []
        for ln in self._lines()[-limit:]:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                out.append({"corrupt_line": True})
        return out

    def verify(self) -> dict:
        prev = GENESIS
        for n, ln in enumerate(self._lines(), start=1):
            try:
                obj = json.loads(ln)
                claimed = obj.pop("hash")
            except (json.JSONDecodeError, KeyError):
                return {"ok": False, "entries": n, "first_bad_entry": n}
            if obj.get("prev_hash") != prev or sha256_text(canonical(obj)) != claimed:
                return {"ok": False, "entries": n, "first_bad_entry": n}
            prev = claimed
        return {"ok": True, "entries": len(self._lines()), "first_bad_entry": None}
