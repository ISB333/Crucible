"""FROZEN inference harness for the CPU speed experiment.

Editability lives in config.py (the `config` region). This module is immutable
spec: the measurement methodology, the target model, and the quality gate.
"""
from __future__ import annotations

from dataclasses import dataclass

TARGET_MODEL = "/home/isb/models/Qwen3.5-9B-Q4_K_M.gguf"
_VPS_CORES = 12


@dataclass(frozen=True)
class Config:
    """Tunable llama-server parameters. The target model is NOT here — it is frozen."""

    draft_model: str | None = None
    n_threads: int = _VPS_CORES
    n_batch: int = 512
    n_ubatch: int = 512
    flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False
    ctx_size: int = 4096
    draft_max: int = 4
    n_concurrent: int = 1
    cache_policy: str = "on"  # "on" | "off"

    def __post_init__(self) -> None:
        if not 1 <= self.n_threads <= _VPS_CORES:
            raise ValueError(f"n_threads must be in [1, {_VPS_CORES}]")
        if self.n_batch < 1:
            raise ValueError("n_batch must be >= 1")
        if self.n_ubatch < 1:
            raise ValueError("n_ubatch must be >= 1")
        if self.ctx_size < 512:
            raise ValueError("ctx_size must be >= 512")
        if self.n_concurrent < 1:
            raise ValueError("n_concurrent must be >= 1")
        if self.cache_policy not in ("on", "off"):
            raise ValueError("cache_policy must be 'on' or 'off'")
        dm = self.draft_max
        if dm < 1:
            dm = 1
        if dm > 16:
            dm = 16
        object.__setattr__(self, "draft_max", dm)

    def to_cli_args(self) -> list[str]:
        args = [
            "--model", TARGET_MODEL,
            "--threads", str(self.n_threads),
            "--batch-size", str(self.n_batch),
            "--ubatch-size", str(self.n_ubatch),
            "--ctx-size", str(self.ctx_size),
            "--parallel", str(self.n_concurrent),
        ]
        if self.flash_attn:
            args.append("--flash-attn")
        if not self.use_mmap:
            args.append("--no-mmap")
        if self.use_mlock:
            args.append("--mlock")
        if self.draft_model:
            args += ["--model-draft", self.draft_model, "--draft-max", str(self.draft_max)]
        if self.cache_policy == "off":
            args.append("--no-context-shift")
        return args