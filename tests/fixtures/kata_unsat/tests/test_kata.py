from problem import double


def test_impossible_a() -> None:
    assert double(2) == 4


def test_impossible_b() -> None:
    assert double(2) == 5  # contradicts test_impossible_a — unsatisfiable by design
