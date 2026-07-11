"""The single editable surface of the experiment. Gemini edits the `config` region only."""
from harness import Config, TARGET_MODEL  # noqa: F401  (TARGET_MODEL frozen, imported for stability)

# crucible:region start name=config
CONFIG = Config(
    draft_model=None,
    n_threads=12,
    n_batch=512,
    n_ubatch=512,
    flash_attn=True,
    use_mmap=True,
    use_mlock=False,
    ctx_size=4096,
    draft_max=4,
    n_concurrent=1,
    cache_policy="on",
)
# crucible:region end