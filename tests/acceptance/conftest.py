import shutil
from pathlib import Path

from crucible.llm import ScriptedSession, ToolCall


def wr(content: str) -> ToolCall:
    return ToolCall(id="1", name="write_region", args={"name": "solution", "content": content})


def copy_fixture(name: str, tmp_path: Path) -> Path:
    src = Path(__file__).parent.parent / "fixtures" / name
    dst = tmp_path / name
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dst


def scripted(scripts_by_worker: dict[int, list[list[ToolCall]]]):
    def factory(worker: int, episode: int) -> ScriptedSession:
        return ScriptedSession(scripts_by_worker.get(worker, [[]]))

    return factory
