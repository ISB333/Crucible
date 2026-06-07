from pathlib import Path

import pytest

from crucible.sandbox import SubprocessSandbox

pytestmark = pytest.mark.unit


def test_runs_command_and_captures_output(tmp_path: Path) -> None:
    res = SubprocessSandbox().run(["echo", "hello"], cwd=tmp_path, timeout_s=10)
    assert res.ok and res.stdout.strip() == "hello"


def test_nonzero_exit_code(tmp_path: Path) -> None:
    res = SubprocessSandbox().run(
        ["python3", "-c", "raise SystemExit(3)"],
        cwd=tmp_path,
        timeout_s=10,
    )
    assert res.exit_code == 3 and not res.ok


def test_timeout_returns_result_not_exception(tmp_path: Path) -> None:
    res = SubprocessSandbox().run(["sleep", "30"], cwd=tmp_path, timeout_s=0.2)
    assert res.timed_out and res.exit_code == 124


def test_missing_binary_returns_result_not_exception(tmp_path: Path) -> None:
    res = SubprocessSandbox().run(["no-such-binary-xyz"], cwd=tmp_path, timeout_s=5)
    assert res.exit_code == 127 and not res.ok


def test_non_executable_file_returns_result_not_exception(tmp_path: Path) -> None:
    script = tmp_path / "noperm.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o644)
    res = SubprocessSandbox().run([str(script)], cwd=tmp_path, timeout_s=5)
    assert res.exit_code == 126 and not res.ok


def test_env_is_minimal(tmp_path: Path) -> None:
    import os

    os.environ["CRUCIBLE_LEAK_TEST"] = "secret"
    try:
        res = SubprocessSandbox().run(["env"], cwd=tmp_path, timeout_s=10)
        assert "CRUCIBLE_LEAK_TEST" not in res.stdout
    finally:
        del os.environ["CRUCIBLE_LEAK_TEST"]
