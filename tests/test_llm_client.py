import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tempo.llm import LLMClient, LLMError, load_dotenv, parse_json_text

KEY = "rc-" + "ab" * 24


class _Server:
    """A tiny local web server that plays back scripted replies and records what it receives."""

    def __init__(self):
        self.script: list[tuple[int, dict, dict]] = []   # (status, headers, json body)
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                outer.requests.append({"path": self.path, "method": self.command, "body": body,
                                       "auth": self.headers.get("Authorization")})
                status, headers, payload = outer.script.pop(0) if outer.script else (200, {}, {})
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _serve

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture()
def server():
    s = _Server()
    yield s
    s.close()


def reply(text, finish="stop"):
    return 200, {}, {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


def make_client(server, sleeps=None, **kw):
    sleeps = sleeps if sleeps is not None else []
    kw.setdefault("rpm", None)
    return LLMClient(base_url=server.url, api_key=KEY, model="test-model", trust_env_proxy=False,
                     sleep=sleeps.append, **kw)


def test_chat_sends_key_model_and_json_mode(server):
    server.script = [reply('{"a": 1}')]
    out = make_client(server).chat_json([{"role": "user", "content": "hi"}])
    assert out == {"a": 1}
    req = server.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == f"Bearer {KEY}"
    assert req["body"]["model"] == "test-model"
    assert req["body"]["response_format"] == {"type": "json_object"}


def test_retries_429_then_succeeds(server):
    server.script = [(429, {"Retry-After": "0"}, {"error": {"message": "slow down"}}), reply("ok")]
    sleeps = []
    assert make_client(server, sleeps).chat([{"role": "user", "content": "hi"}]) == "ok"
    assert len(server.requests) == 2 and len(sleeps) == 1


def test_gives_up_after_repeated_503(server):
    server.script = [(503, {}, {"error": {"message": "busy"}})] * 10
    c = make_client(server, max_retries=2)
    with pytest.raises(LLMError, match="HTTP 503"):
        c.chat([{"role": "user", "content": "hi"}])
    assert len(server.requests) == 3


def test_401_gives_a_helpful_message_and_never_leaks_the_key(server):
    server.script = [(401, {}, {"error": {"message": "invalid key"}})]
    with pytest.raises(LLMError) as e:
        make_client(server).chat([{"role": "user", "content": "hi"}])
    assert "HTTP 401" in str(e.value) and "RADEON_API_KEY" in str(e.value)
    assert KEY not in str(e.value)


def test_platform_style_error_body_is_understood(server):
    server.script = [(400, {}, {"detail": {"error": {"message": "not enough credits"}}})]
    with pytest.raises(LLMError, match="not enough credits"):
        make_client(server).chat([{"role": "user", "content": "hi"}])


def test_empty_reply_from_thinking_gives_a_hint(server):
    server.script = [reply("", finish="length")]
    with pytest.raises(LLMError, match="max_tokens"):
        make_client(server).chat([{"role": "user", "content": "hi"}])


def test_json_in_code_fences_and_recovery_from_non_json(server):
    server.script = [reply("Sure! here you go"), reply('```json\n{"ok": true}\n```')]
    out = make_client(server).chat_json([{"role": "user", "content": "hi"}])
    assert out == {"ok": True} and len(server.requests) == 2
    assert parse_json_text('noise {"x": 2} more noise') == {"x": 2}
    with pytest.raises(LLMError):
        parse_json_text("[1, 2, 3]")


def test_models_lists_ids(server):
    server.script = [(200, {}, {"data": [{"id": "A"}, {"id": "B"}]})]
    assert make_client(server).models() == ["A", "B"]
    assert server.requests[0]["method"] == "GET"


def test_self_throttle_spaces_out_calls(server):
    server.script = [reply("1"), reply("2")]
    sleeps = []
    c = make_client(server, sleeps, rpm=60)       # at most one call per second
    c.chat([{"role": "user", "content": "a"}])
    c.chat([{"role": "user", "content": "b"}])
    assert len(sleeps) == 1 and 0.5 < sleeps[0] <= 1.0


def test_repr_hides_the_key(server):
    assert KEY not in repr(make_client(server)) and "hidden" in repr(make_client(server))


def test_dotenv_reads_values_but_never_overrides(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nTEMPO_TEST_A="quoted value"\nTEMPO_TEST_B=plain\nTEMPO_TEST_C=from_file\n')
    monkeypatch.delenv("TEMPO_TEST_A", raising=False)
    monkeypatch.delenv("TEMPO_TEST_B", raising=False)
    monkeypatch.setenv("TEMPO_TEST_C", "already_set")
    load_dotenv(f)
    import os
    assert os.environ["TEMPO_TEST_A"] == "quoted value" and os.environ["TEMPO_TEST_B"] == "plain"
    assert os.environ["TEMPO_TEST_C"] == "already_set"
    monkeypatch.delenv("TEMPO_TEST_A")
    monkeypatch.delenv("TEMPO_TEST_B")


def test_from_env_explains_a_missing_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RADEON_API_KEY", raising=False)
    with pytest.raises(LLMError, match="RADEON_API_KEY"):
        LLMClient.from_env()


def test_models_accepts_a_bare_list_too(server):
    server.script = [(200, {}, [{"id": "A"}, "B"])]
    assert make_client(server).models() == ["A", "B"]
