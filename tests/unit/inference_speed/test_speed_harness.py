from pathlib import Path

from harness import Config, run_harness  # type: ignore[import-not-found]


def _ws_with_fixtures(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "workload").mkdir(parents=True)
    (ws / "workload" / "prompts_single.jsonl").write_text('{"id":"s1","prompt":"x"}\n')
    (ws / "workload" / "prompts_aggregate.jsonl").write_text(
        '{"id":"a1","prompt":"y"}\n{"id":"a2","prompt":"z"}\n'
    )
    (ws / "workload" / "probes.jsonl").write_text('{"id":"p1","prompt":"q"}\n')
    return ws


def test_run_harness_emits_full_result(tmp_path):
    ws = _ws_with_fixtures(tmp_path)
    cfg = Config()

    def fake_stream(base_url, prompt, max_tokens, temperature=0.0):
        return [("a", 0.0), ("b", 0.5)]  # 2 tok / 0.5s = 4.0 tok/s

    def fake_completion(base_url, prompt, max_tokens):
        return "ANSWER"

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_launcher(c, port):
        return FakeProc(), f"http://127.0.0.1:{port}"

    def fake_waiter(url, timeout_s=120.0):
        return True

    res = run_harness(
        cfg,
        ws,
        stream_fn=fake_stream,
        launcher=fake_launcher,
        waiter=fake_waiter,
        completion_fn=fake_completion,
    )
    assert res["single_stream"]["n_tokens"] == 2
    assert res["aggregate"]["n_tokens"] == 4
    assert res["quality"]["path"] == "lossless"
    assert res["probe_outputs"] == {"p1": "ANSWER"}
    assert "config" in res and res["config"]["n_threads"] == 12
    assert res["loaded_model"] == "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf"


def test_run_harness_returns_error_when_not_ready(tmp_path):
    ws = _ws_with_fixtures(tmp_path)

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_launcher(c, port):
        return FakeProc(), f"http://127.0.0.1:{port}"

    res = run_harness(
        Config(),
        ws,
        stream_fn=lambda *a, **k: [],
        launcher=fake_launcher,
        waiter=lambda url, timeout_s=120.0: False,
        completion_fn=lambda *a, **k: "",
    )
    assert "error" in res
    assert res["error"] == "server did not become ready"
