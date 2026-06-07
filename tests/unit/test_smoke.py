import pytest

import crucible

pytestmark = pytest.mark.unit


def test_package_imports() -> None:
    assert crucible.__version__ == "0.0.1"
