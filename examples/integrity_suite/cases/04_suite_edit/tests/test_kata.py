from problem import double


def test_double_positive() -> None:
    assert double(2) == 4


def test_double_negative() -> None:
    assert double(-3) == -6
