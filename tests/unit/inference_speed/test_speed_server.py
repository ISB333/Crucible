import harness  # type: ignore[import-not-found]
from harness import Config, launch_server, wait_for_ready  # type: ignore[import-not-found]


def test_wait_for_ready_succeeds_on_200():
    calls = {"n": 0}

    def fake_get(url, timeout=1.0):
        calls["n"] += 1

        class R:
            status_code = 200

        return R()

    assert wait_for_ready("http://x:1234/health", timeout_s=2.0, http_get=fake_get) is True
    assert calls["n"] >= 1


def test_wait_for_ready_times_out_on_500(monkeypatch):
    t = [0.0]
    monkeypatch.setattr(
        harness._time, "perf_counter", lambda: t.__setitem__(0, t[0] + 10.0) or t[0]
    )

    def fake_get(url, timeout=1.0):
        class R:
            status_code = 500

        return R()

    assert wait_for_ready("http://x:1234/health", timeout_s=1.0, http_get=fake_get) is False


def test_launch_server_uses_config_cli_args():
    captured = {}

    class FakeProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_runner(cmd, **kw):
        captured["cmd"] = cmd
        return FakeProc()

    proc, base_url = launch_server(
        Config(n_threads=8, n_concurrent=4), port=8080, runner=fake_runner
    )
    assert "--threads" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--threads") + 1] == "8"
    assert base_url == "http://127.0.0.1:8080"
