import pytest
from harness import Config  # type: ignore[import-not-found]


def test_config_defaults_valid():
    c = Config()
    assert c.n_threads == 12
    assert c.draft_model is None
    assert c.cache_policy == "on"


def test_config_rejects_bad_threads():
    with pytest.raises(ValueError, match="n_threads"):
        Config(n_threads=0)
    with pytest.raises(ValueError, match="n_threads"):
        Config(n_threads=99)


def test_config_rejects_bad_batch():
    with pytest.raises(ValueError, match="n_batch"):
        Config(n_batch=0)


def test_config_rejects_bad_cache_policy():
    with pytest.raises(ValueError, match="cache_policy"):
        Config(cache_policy="fast")


def test_config_rejects_bad_concurrency():
    with pytest.raises(ValueError, match="n_concurrent"):
        Config(n_concurrent=0)


def test_config_clamps_draft_max():
    assert Config(draft_max=0).draft_max == 1
    assert Config(draft_max=99).draft_max == 16


def test_config_to_cli_args_has_target_model():
    args = Config().to_cli_args()
    assert "--model" in args
    idx = args.index("--model")
    assert args[idx + 1] == "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf"
