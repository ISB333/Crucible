"""The single editable surface of the experiment. Gemini edits the `config` region only."""

from harness import (  # noqa: F401  (TARGET_MODEL frozen, imported for stability)
    TARGET_MODEL,
    Config,
)

# crucible:region start name=config
# Self-speculative draft: a lower-precision quant of the SAME Qwen3.5-9B (same
# 248320-token tokenizer by construction -> lossless). The Q4 target verifies the
# draft's proposals in one forward pass. DRAFT_Q3 (Q3_K_M, ~4.7GB, high acceptance)
# usually beats DRAFT_Q2 (Q2_K, ~3.9GB, low acceptance) on this setup. Tune
# draft_max (tokens drafted per step; higher = more speedup if acceptance stays
# high) and n_concurrent (batching for aggregate). draft_model=None disables
# spec decoding.
DRAFT_Q2 = "/home/isb/models/Qwen3.5-9B-Q2_K.gguf"
DRAFT_Q3 = "/home/isb/models/Qwen3.5-9B-Q3_K_M.gguf"
CONFIG = Config(
    draft_model=DRAFT_Q2,
    n_threads=12,
    n_batch=512,
    n_ubatch=512,
    flash_attn=True,
    use_mmap=True,
    use_mlock=False,
    ctx_size=4096,
    draft_max=8,
    n_concurrent=1,
    cache_policy="on",
)
# crucible:region end
