from pathlib import Path

import pytest

from crucible.artifact import Artifact, Hole
from crucible.sandbox import Sandbox, SandboxResult
from crucible.verify import Fail, Ok, Partial, RunContext, Span, Verifier, render

pytestmark = pytest.mark.unit


def make_artifact() -> Artifact:
    return Artifact.from_files(
        {"f.py": "# crucible:region start name=s\nx = 1\n# crucible:region end\n"}
    )


def test_render_ok() -> None:
    assert render(Ok(produced=make_artifact())).startswith("VERDICT: OK")


def test_render_partial_lists_holes() -> None:
    v = Partial(
        open_holes=(Hole(file="f.py", line=4, kind="sorry", text="sorry"),),
        feedback="1 goal open",
    )
    text = render(v)
    assert "VERDICT: PARTIAL — 1 open hole(s)" in text
    assert "f.py:5 [sorry] sorry" in text
    assert "1 goal open" in text


def test_render_fail_with_locus() -> None:
    v = Fail(
        feedback="SyntaxError: invalid syntax",
        locus=Span(file="f.py", line_start=2, line_end=2),
    )
    text = render(v)
    assert "VERDICT: FAIL at f.py:3" in text
    assert "SyntaxError" in text


def test_materialize_writes_files_and_is_content_addressed(make_ctx) -> None:
    ctx: RunContext = make_ctx()
    a = make_artifact()
    ws1 = ctx.materialize(a)
    ws2 = ctx.materialize(a)
    assert ws1 == ws2  # same content -> same dir
    assert (ws1 / "f.py").read_text() == a.files["f.py"]


def test_fake_sandbox_satisfies_protocol(make_ctx) -> None:
    ctx: RunContext = make_ctx()
    assert isinstance(ctx.sandbox, Sandbox)
    assert SandboxResult(exit_code=0, stdout="", stderr="").ok


def test_materialize_concurrent_same_artifact_is_safe(make_ctx) -> None:
    import threading

    ctx: RunContext = make_ctx()
    a = make_artifact()
    paths: list[Path] = []
    errors: list[BaseException] = []

    def go() -> None:
        try:
            paths.append(ctx.materialize(a))
        except BaseException as e:  # noqa: BLE001 — the test asserts none occur
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(set(paths)) == 1
    assert (paths[0] / "f.py").read_text() == a.files["f.py"]


def test_materialize_ignores_stale_tmp_leftovers(make_ctx) -> None:
    ctx: RunContext = make_ctx()
    a = make_artifact()
    dst = ctx.scratch / a.content_hash[:16]
    stale = dst.parent / (dst.name + ".tmp")
    stale.mkdir(parents=True)
    (stale / "leftover.py").write_text("junk\n")
    ws = ctx.materialize(a)
    assert sorted(p.name for p in ws.iterdir()) == ["f.py"]


def test_verifier_protocol_structural() -> None:
    class V:
        deterministic = True

        def verify(self, artifact: Artifact, ctx: RunContext) -> Ok:
            return Ok(produced=artifact)

    assert isinstance(V(), Verifier)
