"""Speculative decoding requires draft and target to share a tokenizer.

Checked by the verifier before any --model-draft attempt; an incompatible draft is
a Fail, not a crash. The unit path is seam-tested; the real path compares gguf
metadata (arch + token-table bytes) via the gguf library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def tokenizers_compatible(
    target_gguf: str,
    draft_gguf: str,
    loader: Callable[[str], list[str]] | None = None,
) -> bool:
    """Compare the token sets of two gguf models. loader(path)->list[str] is a seam."""
    if loader is None:
        return _real_compare(target_gguf, draft_gguf)
    return loader(target_gguf) == loader(draft_gguf)


def _real_compare(target_gguf: str, draft_gguf: str) -> bool:
    """Compare tokenizers via gguf metadata: arch + token-table bytes must match.

    Reading the raw token-string blob (gguf field parts) gives an exact comparison
    without launching llama-server. Falls back to False on any read error (safe).
    """
    import gguf

    def sig(path: str) -> tuple | None:
        try:
            r = gguf.GGUFReader(path)
            arch = r.get_field("general.architecture")
            toks = r.get_field("tokenizer.ggml.tokens")
            if arch is None or toks is None:
                return None
            return (bytes(arch.parts[-1]), len(toks.data), bytes(toks.parts[-1]))
        except Exception:
            return None

    return sig(target_gguf) is not None and sig(target_gguf) == sig(draft_gguf)
