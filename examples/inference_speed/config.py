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
#
# LEVERS (the verifier, not confidence, decides what survives):
#   cache_type_k / cache_type_v : KV-cache quant ("f16" default = no flag = baseline;
#       try "q8_0" or "f8" — near-lossless, halves KV read bandwidth; "q4_0" is smaller
#       but the lossless gate rejects it if greedy output diverges). Big lever on long
#       contexts; on these short prompts, expect a small-to-neutral effect — still worth
#       exploring in combination.
#   draft_threads : dedicated threads for the draft model so it stops stealing the
#       target's memory bandwidth during self-spec. None = inherit (default). Try 1-2
#       to keep the draft gentle, or higher if the target has headroom.
#   numa : "distribute" | "shuffle" | "isolate" (default "" = off). Likely neutral on
#       this single-socket VPS — try once, the verifier will score it.
#   n_concurrent : the AGGREGATE lever. Batching N concurrent sequences reads the 5.5GB
#       weight block once for N tokens, so aggregate tok/s can exceed single-stream.
#       Raise it; the verifier measures both single and aggregate separately.
#
# Use the web_search tool to research current llama.cpp CPU techniques (n-gram / prompt
# lookup speculation, KV-cache quant, draft cpu-mask/prio, continuous batching) before
# editing. Ground edits in real methods, not guesses.
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
    cache_type_k="",
    cache_type_v="",
    numa="",
    draft_threads=None,
)
# crucible:region end
