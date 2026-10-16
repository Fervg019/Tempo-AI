import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from tempo.api.audit import AuditLog
from tempo.api.engine import PRESETS, BadRequest, Engine, ExplainService, Params
from tempo.api.server import App, make_handler
from tempo.forecast import QuantileForecaster
from tempo.llm import FakeLLM
from tempo.simulator import generate_run_to_failure


@pytest.fixture(scope="module")
def models():
    return QuantileForecaster(max_iter=60).fit(generate_run_to_failure(80, seed=0))


def wait_done(svc, p, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = svc.status(p)
        if st["status"] == "done":
            return st
        time.sleep(0.05)
    raise AssertionError("explanation job did not finish")


SHORT = Params.parse(PRESETS["short_staffed"])


# ---- settings ----
def test_params_defaults_and_bounds():
    assert Params.parse({}) == Params()
    assert Params.parse({"machines": "8", "parts_delay": "0.25"}).machines == 8
    for bad in ({"machines": 3}, {"machines": 99}, {"machines": "x"}, {"machines": True}, {"machines": 7.5},
                {"parts_delay": 2}, {"fleet": "wild"}, {"tech_weekday": -1}, {"horizon": 100}):
        with pytest.raises(BadRequest):
            Params.parse(bad)


# ---- plan view ----
def test_plan_view_is_complete_and_consistent(models):
    eng = Engine(models)
    v = eng.plan_view(SHORT)
    assert v["validator"]["ok"] and len(v["machines"]) == SHORT.machines and len(v["days"]) == SHORT.horizon
    assert v["days"][0]["label"] == "Mon" and v["days"][5]["label"] == "Sat"
    assert v["days"][5]["technicians"] == SHORT.tech_weekend == 0
    planned = {m["machine_id"]: m["service_day"] for m in v["machines"] if m["service_day"] is not None}
    from_days = {mid: d["day"] for d in v["days"] for mid in d["repairs"]}
    assert planned == from_days
    assert all(d["repairs"] == [] or d["technicians"] >= len(d["repairs"]) for d in v["days"])
    assert v["overview"]["expected_cost_do_nothing"] >= v["overview"]["expected_total_cost"]
    assert v["overview"]["days_with_no_technicians"]
    assert eng.compute(SHORT) is eng.compute(SHORT)                 # cached


def test_reveal_never_beats_the_oracle(models):
    r = Engine(models).reveal(SHORT)
    assert r["oracle"]["cost"] <= r["tempo"]["cost"] and r["oracle"]["cost"] <= r["do_nothing"]["cost"]
    assert r["do_nothing"]["repairs"] == 0 and len(r["true_days_left"]) == SHORT.machines


# ---- AI job ----
def make_service(models, tmp_path, llm="good", **kw):
    llm = FakeLLM("good") if llm == "good" else llm
    return ExplainService(Engine(models), llm, tmp_path / "cache", model_name="fake-model", **kw)


def test_ai_job_runs_in_background_and_is_cached_on_disk(models, tmp_path):
    svc = make_service(models, tmp_path)
    assert svc.status(SHORT)["status"] == "none"
    assert svc.start(SHORT)["status"] == "running"
    r = wait_done(svc, SHORT)["result"]
    assert r["mode"] == "ai" and r["stats"]["first_try_valid_rate"] == 1.0
    scheduled = sum(1 for m in svc.engine.plan_view(SHORT)["machines"] if m["service_day"] is not None)
    assert len(r["work_orders"]) == scheduled
    svc2 = make_service(models, tmp_path, llm=None)                  # a new server, no key: reads the disk cache
    st = svc2.status(SHORT)
    assert st["status"] == "done" and st["result"]["from_cache"] and st["result"]["mode"] == "ai"


def test_no_key_gives_an_instant_plain_version(models, tmp_path):
    svc = make_service(models, tmp_path, llm=None)
    st = svc.start(SHORT)
    assert st["status"] == "done" and st["result"]["mode"] == "fallback" and "No AI key" in st["result"]["notice"]
    assert {o["source"] for o in st["result"]["work_orders"]} == {"template"}
    assert not (tmp_path / "cache").exists()                         # fallback is never cached as if it were AI


def test_limits_fall_back_instead_of_calling_the_api(models, tmp_path):
    llm = FakeLLM("good")
    svc = make_service(models, tmp_path, llm=llm, per_visitor_hour=1)
    svc.start(SHORT, "alice")
    wait_done(svc, SHORT)
    calls = llm.calls
    other = Params.parse({**PRESETS["short_staffed"], "seed": 9})
    st = svc.start(other, "alice")
    assert st["result"]["mode"] == "fallback" and "hourly" in st["result"]["notice"]
    assert llm.calls == calls                                         # no new API calls were made
    st = svc.start(other, "bob")                                      # a different visitor still gets AI
    assert st["status"] in ("running", "done")


def test_daily_cap(models, tmp_path):
    svc = make_service(models, tmp_path, daily_cap=1)
    svc.start(SHORT, "a")
    wait_done(svc, SHORT)
    st = svc.start(Params.parse({**PRESETS["short_staffed"], "seed": 4}), "b")
    assert st["result"]["mode"] == "fallback" and "daily" in st["result"]["notice"]


def test_orders_beyond_the_ai_cap_get_template_orders(models, tmp_path):
    svc = make_service(models, tmp_path, max_orders=2)
    svc.start(SHORT)
    r = wait_done(svc, SHORT)["result"]
    sources = [o["source"] for o in r["work_orders"]]
    assert sources.count("template") == len(sources) - 2 and len(sources) > 2


# ---- audit log ----
def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append({"n": i, "who": "sam"})
    assert log.verify() == {"ok": True, "entries": 3, "first_bad_entry": None}
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace('"who":"sam"', '"who":"eve"')
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    v = log.verify()
    assert not v["ok"] and v["first_bad_entry"] == 2


def test_audit_detects_a_deleted_entry(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.append({"n": i})
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    (tmp_path / "a.jsonl").write_text(lines[0] + "\n" + lines[2] + "\n")
    assert not log.verify()["ok"]


# ---- HTTP ----
@pytest.fixture()
def live(models, tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>hello</h1>")
    eng = Engine(models)
    svc = ExplainService(eng, FakeLLM("good"), tmp_path / "cache", model_name="fake-model")
    app = App(eng, svc, AuditLog(tmp_path / "audit.jsonl"), static)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, app
    httpd.shutdown()


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def jcall(base, path, body=None):
    code, raw, _ = call(base, path, body)
    return code, json.loads(raw)


QS = "seed=3&machines=12&horizon=21&tech_weekday=1&tech_weekend=0&parts_delay=0.4&fleet=stressed"


def test_http_basics_and_validation(live):
    base, _ = live
    code, raw, headers = call(base, "/")
    assert code == 200 and b"hello" in raw and headers["X-Content-Type-Options"] == "nosniff"
    code, h = jcall(base, "/api/health")
    assert code == 200 and h["ai"] == "configured" and "short_staffed" in h["presets"]
    assert jcall(base, "/api/plan?" + QS)[0] == 200
    assert jcall(base, "/api/plan?machines=999")[0] == 400
    assert jcall(base, "/nope")[0] == 404
    assert jcall(base, "/api/explain", {"machines": "x"})[0] == 400
    code, _ = jcall(base, "/api/approve", {"x": "y" * 9000})
    assert code == 400


def test_full_flow_explain_approve_audit(live):
    base, app = live
    body = {"params": PRESETS["short_staffed"], "approver": "Sam Lee", "decision": "approve", "note": "ok"}
    code, err = jcall(base, "/api/approve", body)                    # nothing generated yet
    assert code == 409 and "generate the explanation" in err["error"]
    code, st = jcall(base, "/api/explain", PRESETS["short_staffed"])
    assert code == 200
    t0 = time.time()
    while st["status"] != "done" and time.time() - t0 < 20:
        time.sleep(0.1)
        st = jcall(base, "/api/explain?" + QS)[1]
    assert st["status"] == "done" and st["result"]["mode"] == "ai"
    assert jcall(base, "/api/approve", {**body, "approver": "  "})[0] == 400
    assert jcall(base, "/api/approve", {**body, "decision": "maybe"})[0] == 400
    code, ok = jcall(base, "/api/approve", body)
    assert code == 200 and ok["ok"] and ok["entries"] == 1
    code, aud = jcall(base, "/api/audit")
    e = aud["entries"][0]
    assert aud["chain"]["ok"] and e["approver"] == "Sam Lee" and e["decision"] == "approve"
    assert e["work_orders"] and all(len(o["sha256"]) == 64 for o in e["work_orders"])
    assert e["plan_sha256"] == app.engine.plan_view(Params.parse(PRESETS["short_staffed"]))["plan_sha256"]


def test_a_rejection_does_not_need_an_explanation(live):
    base, _ = live
    code, ok = jcall(base, "/api/approve", {"params": PRESETS["well_staffed"], "approver": "Ana", "decision": "reject",
                                            "note": "wrong week"})
    assert code == 200 and ok["ok"]


def test_server_errors_do_not_leak_details(live):
    base, app = live
    app.engine.plan_view = lambda p: (_ for _ in ()).throw(RuntimeError("secret path /home/x"))
    code, raw, _ = call(base, "/api/plan?" + QS)
    assert code == 500 and b"secret" not in raw


def test_decisions_are_rate_limited_and_the_log_has_a_cap(models, tmp_path):
    eng = Engine(models)
    svc = ExplainService(eng, FakeLLM("good"), tmp_path / "cache", model_name="fake-model")
    app = App(eng, svc, AuditLog(tmp_path / "audit.jsonl"), tmp_path, approvals_per_hour=2, max_log_entries=3)
    body = {"params": PRESETS["well_staffed"], "approver": "Ana", "decision": "reject"}
    assert app.approve(body, "1.2.3.4")["ok"] and app.approve(body, "1.2.3.4")["ok"]
    from tempo.api.server import Conflict
    with pytest.raises(Conflict, match="too many"):
        app.approve(body, "1.2.3.4")
    assert app.approve(body, "5.6.7.8")["ok"]                 # another visitor is fine: this is the 3rd entry
    with pytest.raises(Conflict, match="full"):
        app.approve(body, "9.9.9.9")


def test_saved_text_for_a_different_plan_is_ignored(models, tmp_path):
    svc = make_service(models, tmp_path)
    svc.start(SHORT)
    wait_done(svc, SHORT)
    path = svc._disk_path(svc.key(SHORT))
    saved = json.loads(path.read_text())
    saved["plan_sha256"] = "0" * 64                                   # as if it was written for another plan
    path.write_text(json.dumps(saved))
    svc2 = make_service(models, tmp_path, llm=None)
    assert svc2.status(SHORT)["status"] == "none"
