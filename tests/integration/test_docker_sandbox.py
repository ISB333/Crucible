import shutil
from pathlib import Path

import pytest

from crucible.sandbox import DockerSandbox

pytestmark = pytest.mark.integration  # needs the Docker daemon + python:3.12-slim pulled


def test_runs_in_container(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("print('from-container')\n")
    res = DockerSandbox().run(["python3", "hello.py"], cwd=tmp_path, timeout_s=60)
    assert res.ok and res.stdout.strip() == "from-container"


def test_network_disabled_by_default(tmp_path: Path) -> None:
    code = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=2)"
    res = DockerSandbox().run(["python3", "-c", code], cwd=tmp_path, timeout_s=60)
    assert not res.ok  # no route — network=none


def test_timeout_kills_container(tmp_path: Path) -> None:
    res = DockerSandbox().run(["sleep", "60"], cwd=tmp_path, timeout_s=2)
    assert res.timed_out and res.exit_code == 124


def test_workspace_writes_are_host_owned(tmp_path: Path) -> None:
    code = "import os; os.makedirs('/work/sub'); open('/work/sub/f.txt', 'w').write('x')"
    res = DockerSandbox().run(["python3", "-c", code], cwd=tmp_path, timeout_s=60)
    assert res.ok
    shutil.rmtree(tmp_path / "sub")  # must not raise PermissionError (root-owned dirs would)
