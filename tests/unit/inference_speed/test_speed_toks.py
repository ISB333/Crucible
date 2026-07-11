from harness import tok_per_second  # type: ignore[import-not-found]


def test_tps_basic():
    # 4 tokens over 1.0s span -> 4.0 tok/s
    events = [("a", 0.0), ("b", 0.3), ("c", 0.7), ("d", 1.0)]
    assert tok_per_second(events) == 4.0


def test_tps_one_event_is_zero():
    assert tok_per_second([("a", 1.0)]) == 0.0


def test_tps_empty_is_zero():
    assert tok_per_second([]) == 0.0


def test_tps_zero_span_is_zero():
    assert tok_per_second([("a", 5.0), ("b", 5.0)]) == 0.0