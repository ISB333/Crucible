import json
from pathlib import Path

from harness import Config  # type: ignore[import-not-found]
from speed_verifier import SpeedQualityVerifier  # type: ignore[import-not-found]

from crucible.artifact import Artifact
from crucible.verify import Fail, Ok, Partial, Scored

_FILLED_CONFIG = (
    "from harness import Config\n"
    "# crucible:region start name=config\n"
    "CONFIG = Config()\n"
    "# crucible:region end\n"
)

_HOLED_CONFIG = (
    "from harness import Config\n"
    "# crucible:region start name=config\n"
    "CONFIG = crucible:hole  # placeholder left by the agent\n"
    "# crucible:region end\n"
)


def _baseline(tmp_path: Path, **extra) -> Path:
    data = {"single_stream": 2.6, "aggregate": 2.6, "probe_reference": {"p1": "ANSWER"}, **extra}
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(data))
    return p


def _artifact(config_text: str) -> Artifact:
    return Artifact.from_files(
        {
            "config.py": config_text,
            "harness.py": "from dataclasses import dataclass\n",  # frozen stub (loader bypassed)
            "strategy.py": "",
        }
    )


def _result(agg: float, single: float, probe="ANSWER", loaded=None):
    from harness import TARGET_MODEL  # type: ignore[import-not-found]

    return {
        "aggregate": {"tok_s": agg},
        "single_stream": {"tok_s": single},
        "quality": {"path": "lossless", "match": None, "mismatched": None},
        "loaded_model": loaded or TARGET_MODEL,
        "probe_outputs": {"p1": probe},
        "config": {},
    }


def test_partial_when_config_hole_sentinel(tmp_path, make_ctx):
    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: _result(40.0, 9.0),
        config_loader=lambda ws: Config(),
    )
    verdict = v.verify(_artifact(_HOLED_CONFIG), make_ctx())
    assert isinstance(verdict, Partial)


def test_ok_when_targets_met_and_quality_clean(tmp_path, make_ctx):
    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: _result(31.0, 9.0),
        config_loader=lambda ws: Config(),
    )
    verdict = v.verify(_artifact(_FILLED_CONFIG), make_ctx())
    assert isinstance(verdict, Ok)


def test_scored_when_quality_clean_but_below_target(tmp_path, make_ctx):
    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: _result(10.0, 4.0),
        config_loader=lambda ws: Config(),
    )
    verdict = v.verify(_artifact(_FILLED_CONFIG), make_ctx())
    assert isinstance(verdict, Scored)
    assert verdict.value == 10.0


def test_fail_when_quality_mismatch(tmp_path, make_ctx):
    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: _result(40.0, 9.0, probe="WRONG"),
        config_loader=lambda ws: Config(),
    )
    verdict = v.verify(_artifact(_FILLED_CONFIG), make_ctx())
    assert isinstance(verdict, Fail)


def test_fail_when_wrong_model_loaded(tmp_path, make_ctx):
    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: _result(40.0, 9.0, loaded="/home/isb/models/tiny.gguf"),
        config_loader=lambda ws: Config(),
    )
    verdict = v.verify(_artifact(_FILLED_CONFIG), make_ctx())
    assert isinstance(verdict, Fail)
    assert "gaming vector" in verdict.feedback


def test_fail_when_harness_error(tmp_path, make_ctx):
    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: {"error": "server did not become ready"},
        config_loader=lambda ws: Config(),
    )
    verdict = v.verify(_artifact(_FILLED_CONFIG), make_ctx())
    assert isinstance(verdict, Fail)


def test_fail_when_config_load_raises(tmp_path, make_ctx):
    def bad_loader(ws):
        raise SyntaxError("broken config")

    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: _result(40.0, 9.0),
        config_loader=bad_loader,
    )
    verdict = v.verify(_artifact(_FILLED_CONFIG), make_ctx())
    assert isinstance(verdict, Fail)
    assert "valid Config" in verdict.feedback


def test_fail_when_draft_tokenizer_incompatible(tmp_path, make_ctx):
    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: _result(40.0, 9.0),
        config_loader=lambda ws: Config(draft_model="/some/draft.gguf"),
        compat_checker=lambda target, draft: False,
    )
    verdict = v.verify(_artifact(_FILLED_CONFIG), make_ctx())
    assert isinstance(verdict, Fail)
    assert "incompatible" in verdict.feedback


def test_draft_compatible_proceeds_to_runner(tmp_path, make_ctx):
    v = SpeedQualityVerifier(
        baseline_path=_baseline(tmp_path),
        runner=lambda cfg, ws: _result(31.0, 9.0),
        config_loader=lambda ws: Config(draft_model="/some/draft.gguf"),
        compat_checker=lambda target, draft: True,
    )
    verdict = v.verify(_artifact(_FILLED_CONFIG), make_ctx())
    assert isinstance(verdict, Ok)


def test_task_from_path_marks_only_config_editable():
    from crucible import Task

    example = Path(__file__).resolve().parents[3] / "examples" / "inference_speed"
    t = Task.from_path(example, editable=["config"], network=True)
    assert "config" in t.editable
    assert "harness.py" in t.files
    assert "workload/prompts_single.jsonl" in t.files
    assert "strategy.py" in t.files
    assert t.network is True
