"""FROZEN inference harness for the CPU speed experiment.

Editability lives in config.py (the `config` region). This module is immutable
spec: the measurement methodology, the target model, and the quality gate.
"""

from __future__ import annotations

import json as _json
import subprocess as _subprocess
import time as _time
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    class _Proc(Protocol):
        """Minimal subprocess-like handle the harness needs to stop a server."""

        def terminate(self) -> None: ...
        def wait(self, timeout: float | None = ...) -> object: ...


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
            "--model",
            TARGET_MODEL,
            "--threads",
            str(self.n_threads),
            "--batch-size",
            str(self.n_batch),
            "--ubatch-size",
            str(self.n_ubatch),
            "--ctx-size",
            str(self.ctx_size),
            "--parallel",
            str(self.n_concurrent),
        ]
        if self.flash_attn:
            args += ["--flash-attn", "on"]  # --flash-attn takes a value (on|off|auto)
        if not self.use_mmap:
            args.append("--no-mmap")
        if self.use_mlock:
            args.append("--mlock")
        if self.draft_model:
            args += [
                "--model-draft",
                self.draft_model,
                "--spec-draft-n-max",
                str(self.draft_max),
            ]
        if self.cache_policy == "off":
            args.append("--no-context-shift")
        return args


# --- measurement core (seam-injected; never loads the 9B in unit tests) ---


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


def measure_single_stream(
    stream_fn, base_url: str, prompts: list[dict], max_tokens: int = 256
) -> dict:
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


def measure_aggregate(
    stream_fn,
    base_url: str,
    prompts: list[dict],
    max_tokens: int = 256,
    max_workers: int | None = None,
) -> dict:
    """Fire prompts concurrently; aggregate tok/s = total_tokens / wall_clock.

    max_workers caps concurrency (should be config.n_concurrent so we never queue
    more requests than llama-server has slots — queued requests time out). None ->
    len(prompts) (back-compat). Wall clock is real perf_counter.
    """
    import concurrent.futures as cf

    def one(item):
        return stream_fn(base_url, item["prompt"], max_tokens, temperature=0.0)

    workers = max_workers if max_workers is not None else max(1, len(prompts))
    t0 = _time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, prompts))
    t1 = _time.perf_counter()
    wall = t1 - t0
    total = sum(len(r) for r in results)
    tps = total / wall if wall > 0 else 0.0
    return {"tok_s": tps, "n_tokens": total, "wall_s": wall}


def lossless_match(
    probe_outputs: dict[str, str], reference: dict[str, str]
) -> tuple[bool, list[str]]:
    """Byte-identical comparison of candidate probe outputs vs frozen reference.

    For lossless-by-construction optimizations (spec decoding, batching, cache),
    accepted output MUST equal the greedy reference. Any mismatch is a quality
    regression -> the verifier returns Fail.
    """
    mism = [pid for pid in reference if probe_outputs.get(pid) != reference[pid]]
    return (len(mism) == 0, mism)


# --- server lifecycle + real streamer (integration; seams for unit tests) ---


def wait_for_ready(base_url: str, timeout_s: float = 120.0, http_get=None) -> bool:
    """Poll {base_url}/health until 200 or timeout. http_get is a seam (default requests.get)."""
    if http_get is None:
        import requests

        http_get = requests.get
    deadline = _time.perf_counter() + timeout_s
    url = base_url.rstrip("/") + "/health"
    while _time.perf_counter() < deadline:
        try:
            r = http_get(url, timeout=2.0)
            if getattr(r, "status_code", 0) == 200:
                return True
        except Exception:
            pass
        _time.sleep(0.5)
    return False


def _free_port() -> int:
    """Pick an ephemeral free port (small TOCTOU race — acceptable for per-attempt servers)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_server(
    config: Config,
    port: int | None = None,
    runner: Callable[..., _Proc] | None = None,
) -> tuple[_Proc, str]:
    """Start llama-server with config.to_cli_args(). port=None -> auto-pick a free port.

    runner is a seam (default subprocess.Popen). Auto port lets parallel verifier
    calls each get their own server without colliding.
    """
    if runner is None:
        runner = _subprocess.Popen  # Popen is structurally _Proc (terminate + wait)
    if port is None:
        port = _free_port()
    cmd = ["llama-server", *config.to_cli_args(), "--port", str(port), "--host", "127.0.0.1"]
    # DEVNULL (not PIPE): an unread PIPE would fill and block llama-server on heavy logging.
    # To debug a launch failure, run llama-server manually with the same to_cli_args().
    proc = runner(cmd, stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL)
    return proc, f"http://127.0.0.1:{port}"


def httpx_stream(
    base_url: str, prompt: str, max_tokens: int, temperature: float = 0.0
) -> list[tuple[str, float]]:
    """Real streamer: POST {base_url}/v1/chat/completions with stream=true.

    Records perf_counter per emitted chunk -> list[(token, timestamp)].
    """
    import httpx

    body = {
        "model": "target",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Qwen3.5 is a reasoning model: without this it emits delta.reasoning_content
        # (thinking) before delta.content, and a small max_tokens never reaches an answer.
        # Disabling thinking makes it a fast direct answerer — the coding-agent use case.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    events: list[tuple[str, float]] = []
    with httpx.Client(timeout=300.0) as client:
        with client.stream(
            "POST", base_url.rstrip("/") + "/v1/chat/completions", json=body
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = _json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    # content is the answer; fall back to reasoning_content/token so
                    # throughput is still counted if a model can't disable thinking.
                    tok = delta.get("content") or delta.get("reasoning_content") or delta.get("token") or ""
                except Exception:
                    continue
                if tok:
                    events.append((tok, _time.perf_counter()))
    return events


# --- orchestration ---


def _join_completion(stream_fn):
    """Wrap a stream_fn into a non-streaming greedy completion: returns joined text."""

    def completion(base_url, prompt, max_tokens):
        events = stream_fn(base_url, prompt, max_tokens, temperature=0.0)
        return "".join(tok for tok, _ in events)

    return completion


def _config_dict(c: Config) -> dict:
    from dataclasses import asdict

    return asdict(c)


def run_harness(
    config: Config,
    workspace: _Path,
    stream_fn=None,
    launcher=None,
    waiter=None,
    completion_fn=None,
    max_tokens: int = 256,
) -> dict:
    """Launch the target, measure single + aggregate, run the lossless probe check.

    Returns the JSON result dict. All 9B interaction is via seams; unit tests pass
    fakes. The workspace must contain workload/prompts_single.jsonl,
    workload/prompts_aggregate.jsonl, workload/probes.jsonl. The lossless reference
    comparison is done by the verifier (which holds baseline.json); the harness
    returns raw probe_outputs and a placeholder quality block.
    """
    stream_fn = stream_fn or httpx_stream
    launcher = launcher or (lambda c, port: launch_server(c, port))
    waiter = waiter or (lambda url, timeout_s=120.0: wait_for_ready(url, timeout_s))
    completion_fn = completion_fn or _join_completion(stream_fn)

    proc, base_url = launcher(config, None)  # None -> launch_server auto-picks a free port
    try:
        if not waiter(base_url):
            return {"error": "server did not become ready", "config": _config_dict(config)}
        single_prompts = load_workload(workspace / "workload" / "prompts_single.jsonl")
        agg_prompts = load_workload(workspace / "workload" / "prompts_aggregate.jsonl")
        probes = load_workload(workspace / "workload" / "probes.jsonl")

        single = measure_single_stream(stream_fn, base_url, single_prompts, max_tokens=max_tokens)
        aggregate = measure_aggregate(
            stream_fn, base_url, agg_prompts, max_tokens=max_tokens, max_workers=config.n_concurrent
        )

        probe_outputs = {p["id"]: completion_fn(base_url, p["prompt"], 16) for p in probes}
        return {
            "config": _config_dict(config),
            "single_stream": single,
            "aggregate": aggregate,
            "probe_outputs": probe_outputs,
            "quality": {
                "path": "lossless",
                "match": None,
                "mismatched": None,
            },  # filled by verifier
            "loaded_model": TARGET_MODEL,
        }
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            pass
