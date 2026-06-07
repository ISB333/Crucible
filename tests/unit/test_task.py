from pathlib import Path

import pytest

from crucible.task import Task

pytestmark = pytest.mark.unit


def test_from_single_file(tmp_path: Path) -> None:
    f = tmp_path / "problem.py"
    f.write_text("x = 1\n")
    task = Task.from_path(f, editable=["solution"])
    assert task.root == tmp_path
    assert task.files == ("problem.py",)
    assert task.editable == ("solution",)
    assert task.network is False


def test_from_directory_sorted_and_skips_hidden(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("b\n")
    (tmp_path / "a.py").write_text("a\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("c\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("hidden\n")
    task = Task.from_path(tmp_path, editable=["solution"])
    assert task.files == ("a.py", "b.py", "sub/c.txt")


def test_load_files_reads_contents(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('hi')\n")
    task = Task.from_path(tmp_path, editable=["s"])
    assert task.load_files() == {"a.py": "print('hi')\n"}
