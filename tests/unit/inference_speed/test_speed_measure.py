from harness import (  # type: ignore[import-not-found]
    load_workload,
    measure_aggregate,
    measure_single_stream,
)


def _fake_stream(emit_per_second: float = 10.0):
    """stream_fn that emits n tokens at a fixed cadence (deterministic, no real time)."""

    def stream_fn(base_url, prompt, max_tokens, temperature=0.0):
        n = 20
        return [(f"tok{i}", i / emit_per_second) for i in range(n)]

    return stream_fn


def test_load_workload(tmp_path):
    p = tmp_path / "w.jsonl"
    p.write_text('{"id":"x","prompt":"hi"}\n{"id":"y","prompt":"yo"}\n')
    items = load_workload(p)
    assert [i["id"] for i in items] == ["x", "y"]


def test_measure_single_stream_aggregates_over_prompts():
    prompts = [{"id": "s1", "prompt": "x"}, {"id": "s2", "prompt": "y"}]
    res = measure_single_stream(_fake_stream(10.0), "http://x", prompts, max_tokens=20)
    # each prompt: 20 tokens over 1.9s span (i=0..19 -> 0..1.9) -> 20/1.9 ~ 10.53
    assert res["n_tokens"] == 40
    assert res["tok_s"] > 0.0
    assert len(res["per_prompt"]) == 2
    assert res["per_prompt"][0]["id"] == "s1"


def test_measure_single_stream_empty_prompts():
    res = measure_single_stream(_fake_stream(), "http://x", [], max_tokens=20)
    assert res["n_tokens"] == 0
    assert res["tok_s"] == 0.0


def test_measure_aggregate_total_tokens_and_positive_throughput():
    prompts = [{"id": f"a{i}", "prompt": str(i)} for i in range(4)]

    def stream_fn(base_url, prompt, max_tokens, temperature=0.0):
        return [(f"t{j}", j / 100.0) for j in range(10)]

    res = measure_aggregate(stream_fn, "http://x", prompts, max_tokens=10)
    assert res["n_tokens"] == 40
    assert res["tok_s"] > 0.0
    assert res["wall_s"] >= 0.0
