import os

import pytest

from crucible.providers import make_session

pytestmark = pytest.mark.integration


@pytest.mark.skipif("ANTHROPIC_API_KEY" not in os.environ, reason="no API key")
def test_anthropic_session_round_trip() -> None:
    s = make_session("claude-haiku-4-5-20251001")
    calls = s.start(
        "You are a test. Call the write_region tool exactly once with name='s', content='x = 1'.",
        "Do it now.",
    )
    assert calls and calls[0].name == "write_region"
    assert s.cost_usd > 0
