from check_draft_compat import tokenizers_compatible  # type: ignore[import-not-found]


def test_compatible_when_same_vocab():
    def fake_tokens(path):
        return ["hello", "world", " "]

    assert tokenizers_compatible("/a", "/b", loader=fake_tokens) is True


def test_incompatible_when_vocab_differs():
    seq = iter([["hello", "world"], ["hello", "WORLD"]])

    def fake_tokens(path):
        return next(seq)

    assert tokenizers_compatible("/a", "/b", loader=fake_tokens) is False
