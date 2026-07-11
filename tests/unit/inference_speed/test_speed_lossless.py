from harness import lossless_match  # type: ignore[import-not-found]


def test_all_match():
    ref = {"p1": "Paris", "p2": "4"}
    out = {"p1": "Paris", "p2": "4"}
    ok, mism = lossless_match(out, ref)
    assert ok and mism == []


def test_one_mismatch():
    ref = {"p1": "Paris", "p2": "4"}
    out = {"p1": "paris", "p2": "4"}
    ok, mism = lossless_match(out, ref)
    assert not ok and mism == ["p1"]


def test_missing_probe_is_mismatch():
    ref = {"p1": "Paris", "p2": "4"}
    out = {"p1": "Paris"}
    ok, mism = lossless_match(out, ref)
    assert not ok and "p2" in mism