"""Web server for the supervisor screen (Python standard library only, nothing extra to install).

    python -m tempo.api.server                 http://127.0.0.1:8000
    python -m tempo.api.server --warm          pre-generate the AI explanations for the demo scenarios, then exit

The API key stays on the server (it is read from .env here and never sent to the browser).
Settings: TEMPO_HOST (default 127.0.0.1), TEMPO_PORT or PORT (default 8000), TEMPO_DAILY_AI (default 40
AI generations per day), TEMPO_VISITOR_AI (default 6 per visitor per hour), TEMPO_TRUST_PROXY=1 to use the
X-Forwarded-For address when running behind a proxy.
"""
from __future__ import annotations

import argparse
import json
import threading
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tempo.llm import LLMClient, LLMError, load_dotenv

from .audit import AuditLog, sha256_text, canonical
from .engine import PRESETS, BadRequest, Engine, ExplainService, Params

ROOT = Path(__file__).resolve().parents[2]
MAX_BODY = 8_000
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def load_models(cache_path: Path):
    """Train the forecaster (about 10 seconds) or reuse the copy saved last time."""
    import joblib
    try:
        if cache_path.is_file():
            return joblib.load(cache_path)[1]
    except Exception:  # noqa: BLE001 - a stale or unreadable cache just means "train again"
        pass
    from tempo.planner.demo import build_models
    print("training the forecaster (about 10 seconds) ...", flush=True)
    base, cal = build_models()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((base, cal), cache_path)
    except OSError:
        pass
    return cal


class App:
    """Everything the request handler needs, kept together so tests can build one with fakes."""

    def __init__(self, engine: Engine, explain: ExplainService, audit: AuditLog, static_dir: Path,
                 trust_proxy: bool = False, approvals_per_hour: int = 20, max_log_entries: int = 5000):
        self.engine, self.explain, self.audit = engine, explain, audit
        self.static_dir, self.trust_proxy = static_dir, trust_proxy
        self.approvals_per_hour, self.max_log_entries = approvals_per_hour, max_log_entries
        self._decisions: dict[str, list[float]] = {}
        self._decisions_lock = threading.Lock()

    # ---- API calls (pure functions of the request, easy to test) ----
    def health(self) -> dict:
        return {"ok": True, "ai": "configured" if self.explain.llm is not None else "offline",
                "model": self.explain.model_name if self.explain.llm is not None else None,
                "presets": PRESETS}

    def _check_decision_limits(self, visitor: str) -> None:
        now = time.time()
        with self._decisions_lock:
            hits = [t for t in self._decisions.get(visitor, []) if now - t < 3600]
            if len(hits) >= self.approvals_per_hour:
                self._decisions[visitor] = hits
                raise Conflict("too many decisions from this connection in the last hour, try again later")
            if self.audit.count() >= self.max_log_entries:
                raise Conflict("the decision log of this demo is full")
            hits.append(now)
            self._decisions[visitor] = hits

    def approve(self, body: dict, visitor: str = "local") -> dict:
        p = Params.parse(body.get("params") or {})
        approver = _CONTROL.sub("", str(body.get("approver", ""))).strip()
        note = _CONTROL.sub(" ", str(body.get("note", ""))).strip()[:300]
        decision = body.get("decision")
        if not 1 <= len(approver) <= 60:
            raise BadRequest("enter the approver's name (1 to 60 characters)")
        if decision not in ("approve", "reject"):
            raise BadRequest("decision must be 'approve' or 'reject'")
        self._check_decision_limits(visitor)
        c = self.engine.compute(p)
        st = self.explain.status(p)
        if decision == "approve":
            if not c.report.ok:
                raise Conflict("the plan has rule errors and cannot be approved")
            if st["status"] != "done":
                raise Conflict("generate the explanation and work orders first, so you approve what you have read")
        result = st.get("result") if st["status"] == "done" else None
        orders = result["work_orders"] if result else []
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "approver": approver, "decision": decision,
            "note": note, "params": p.as_dict(), "plan": c.plan_rows(), "plan_sha256": c.plan_hash(),
            "expected_cost": int(round(c.plan.expected_cost)),
            "validator": {"ok": c.report.ok, "warnings": [f.message for f in c.report.warnings]},
            "explanation_mode": result["mode"] if result else None,
            "explanation_text_sha256": sha256_text(canonical([result["summary"], result["reasons"]])) if result else None,
            "work_orders": [{"machine_id": o["machine_id"], "service_day": o["service_day"], "source": o["source"],
                             "sha256": sha256_text(canonical(o))} for o in orders],
        }
        saved = self.audit.append(entry)
        return {"ok": True, "hash": saved["hash"], "entries": self.audit.verify()["entries"]}


class Conflict(RuntimeError):
    pass


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Tempo"

        def log_message(self, fmt, *args):  # quiet; never log query strings or bodies
            return

        # ---- helpers ----
        def _visitor(self) -> str:
            if app.trust_proxy:
                fwd = self.headers.get("X-Forwarded-For", "")
                if fwd:
                    return fwd.split(",")[0].strip()[:64]
            return self.client_address[0]

        def _send(self, status: int, payload, content_type="application/json; charset=utf-8"):
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise BadRequest("request too large")
            try:
                obj = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                raise BadRequest("body is not valid JSON") from None
            if not isinstance(obj, dict):
                raise BadRequest("body must be a JSON object")
            return obj

        def _guard(self, fn):
            try:
                self._send(200, fn())
            except BadRequest as e:
                self._send(400, {"error": str(e)})
            except Conflict as e:
                self._send(409, {"error": str(e)})
            except Exception:  # noqa: BLE001 - never leak internals to the browser
                self._send(500, {"error": "something went wrong on the server"})

        # ---- routes ----
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path in ("/", "/index.html"):
                page = app.static_dir / "index.html"
                if not page.is_file():
                    return self._send(404, {"error": "app/index.html is missing"})
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            routes = {
                "/api/health": lambda: app.health(),
                "/api/plan": lambda: app.engine.plan_view(Params.parse(q)),
                "/api/reveal": lambda: app.engine.reveal(Params.parse(q)),
                "/api/explain": lambda: app.explain.status(Params.parse(q)),
                "/api/audit": lambda: {"entries": app.audit.entries(50), "chain": app.audit.verify()},
            }
            if url.path in routes:
                return self._guard(routes[url.path])
            self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/explain":
                return self._guard(lambda: app.explain.start(Params.parse(self._body()), self._visitor()))
            if path == "/api/approve":
                return self._guard(lambda: app.approve(self._body(), self._visitor()))
            self._send(404, {"error": "not found"})

    return Handler


def build_app(*, models=None, llm="env", audit_path=None, cache_dir=None) -> App:
    load_dotenv(ROOT / ".env")
    if models is None:
        models = load_models(ROOT / ".cache" / "models.joblib")
    if llm == "env":
        try:
            llm = LLMClient.from_env()
        except LLMError:
            llm = None
    engine = Engine(models)
    explain = ExplainService(engine, llm, cache_dir or ROOT / "demo_cache",
                             model_name=getattr(llm, "model", "none") if llm is not None else "none",
                             daily_cap=int(os.environ.get("TEMPO_DAILY_AI", "40")),
                             per_visitor_hour=int(os.environ.get("TEMPO_VISITOR_AI", "6")))
    return App(engine, explain, AuditLog(audit_path or ROOT / "data" / "audit.jsonl"), ROOT / "app",
               trust_proxy=os.environ.get("TEMPO_TRUST_PROXY") == "1")


def warm(app: App) -> int:
    """Generate and save the AI text for the demo scenarios, so the live demo is instant and does not
    depend on the API being up."""
    if app.explain.llm is None:
        print("No API key: cannot warm the cache. Put RADEON_API_KEY in .env first.")
        return 1
    for name, raw in PRESETS.items():
        p = Params.parse(raw)
        print(f"{name}: ", end="", flush=True)
        st = app.explain.start(p, "warm")
        while st["status"] == "running":
            time.sleep(1.0)
            st = app.explain.status(p)
        r = st["result"]
        print(f"{r['mode']}, first-try valid {r['stats']['first_try_valid_rate']:.0%}, "
              f"{r['llm_calls']} calls" + (f"  NOTE: {r['notice']}" if r["notice"] else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("TEMPO_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("TEMPO_PORT") or os.environ.get("PORT") or 8000))
    ap.add_argument("--warm", action="store_true", help="pre-generate AI text for the demo scenarios and exit")
    args = ap.parse_args()
    app = build_app()
    if args.warm:
        return warm(app)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    h = app.health()
    print(f"Tempo is running at http://{args.host}:{args.port}   (AI: {h['ai']}"
          + (f", model {h['model']}" if h["model"] else "") + ")   Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
