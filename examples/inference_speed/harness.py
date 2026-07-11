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


# --- measurement core (seam-injected; never loads the 9B in unit tests) ---

import json as _json
import time as _time
from pathlib import Path as _Path


def tok_per_second(events: list[tuple[str, float]]) -> float:
    """tokens / (last_timestamp - first_timestamp). 0.0 if undefined."""
    if len(events) < 2:
        return 0.0
    t0 = events[0][1]
    t1 = events[-1][1]
    span = t1 - t0
    if span <= 0.0:
        return 0.0
    return len(events) / span


def load_workload(path: _Path) -> list[dict]:
    """Load a JSONL workload file -> list of {"id","prompt"} dicts."""
    items: list[dict] = []
    for line in _Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            items.append(_json.loads(line))
    return items


def measure_single_stream(stream_fn, base_url: str, prompts: list[dict], max_tokens: int = 256) -> dict:
    """Stream each prompt greedily; report median tok/s and totals.

    stream_fn(base_url, prompt, max_tokens, temperature=0.0) -> list[(token, perf_counter)].
    """
    per_prompt: list[dict] = []
    total_tokens = 0
    tps_values: list[float] = []
    for item in prompts:
        events = stream_fn(base_url, item["prompt"], max_tokens, temperature=0.0)
        tps = tok_per_second(events)
        per_prompt.append({"id": item["id"], "tok_s": tps, "n_tokens": len(events)})
        total_tokens += len(events)
        if tps > 0:
            tps_values.append(tps)
    tps_values.sort()
    median = tps_values[len(tps_values) // 2] if tps_values else 0.0
    return {"tok_s": median, "n_tokens": total_tokens, "per_prompt": per_prompt}


def measure_aggregate(stream_fn, base_url: str, prompts: list[dict], max_tokens: int = 256) -> dict:
    """Fire all prompts concurrently; aggregate tok/s = total_tokens / wall_clock.

    The stream_fn is run concurrently via threads (llama-server handles parallel
    decoders when --parallel >= n_concurrent). Wall clock is real perf_counter.
    """
    import concurrent.futures as cf

    def one(item):
        return stream_fn(base_url, item["prompt"], max_tokens, temperature=0.0)

    t0 = _time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
        results = list(pool.map(one, prompts))
    t1 = _time.perf_counter()
    wall = t1 - t0
    total = sum(len(r) for r in results)
    tps = total / wall if wall > 0 else 0.0
    return {"tok_s": tps, "n_tokens": total, "wall_s": wall}


def lossless_match(probe_outputs: dict[str, str], reference: dict[str, str]) -> tuple[bool, list[str]]:
    """Byte-identical comparison of candidate probe outputs vs frozen reference.

    For lossless-by-construction optimizations (spec decoding, batching, cache),
    accepted output MUST equal the greedy reference. Any mismatch is a quality
    regression -> the verifier returns Fail.
    """
    mism = [pid for pid in reference if probe_outputs.get(pid) != reference[pid]]
    return (len(mism) == 0, mism)