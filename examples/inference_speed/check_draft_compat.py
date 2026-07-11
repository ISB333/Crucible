"""Speculative decoding requires draft and target to share a tokenizer.

Checked by the verifier before any --model-draft attempt; an incompatible draft is
a Fail, not a crash. The unit path is seam-tested; the real loader (llama-server
--vocab-only dump) is an integration concern verified at run time.
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
    """Integration path: dump each model's vocab via llama-server and compare token lists.

    The exact stdout format is verified at integration time; if llama-server's
    --vocab-only output changes, adjust the prefix filter here.
    """
    import subprocess

    def dump(path: str) -> list[str]:
        r = subprocess.run(
            ["llama-server", "--model", path, "--vocab-only", "--log-disable"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        lines = (r.stdout + r.stderr).splitlines()
        return [ln for ln in lines if ln.startswith("tok: ")]

    return dump(target_gguf) == dump(draft_gguf)
