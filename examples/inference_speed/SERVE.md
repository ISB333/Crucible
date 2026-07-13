# Using the 9B with the search's best config

The verifier-grounded search found a lossless config that's **3.6× faster
single-stream / 5.2× faster aggregate** than the baseline (no quality change —
byte-identical greedy output, verified 61/61). `serve_best.sh` serves the model
with that config.

## Start the server

```bash
examples/inference_speed/serve_best.sh            # single-stream profile — best for one user
examples/inference_speed/serve_best.sh --agg      # aggregate profile — best for many users
examples/inference_speed/serve_best.sh --port 9090
```

It launches `llama-server` **detached** (`setsid`+`nohup` — survives you logging
out or closing your laptop; the VPS keeps it running) and waits for the server to
be ready. The endpoint is **OpenAI-compatible**:

- URL: `http://<vps-ip>:9090/v1`  (or `http://127.0.0.1:9090/v1` on the box)
- Model name: `qwen`
- Port 9090 is chosen because 8080/8081 are already taken by other services here.

## Use it

**curl:**
```bash
curl http://127.0.0.1:9090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi in 3 words"}],"max_tokens":20,"chat_template_kwargs":{"enable_thinking":false}}'
```

**Python (openai SDK):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9090/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Write a Python fib function."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

**Any tool that takes an OpenAI base URL:** set
`OPENAI_BASE_URL=http://127.0.0.1:9090/v1` (or `http://<vps-ip>:9090/v1` from
another machine) and `OPENAI_API_KEY=anything`.

> **Required:** always pass `chat_template_kwargs.enable_thinking=false`. Qwen3.5 is
> a reasoning model — without this it emits thinking tokens first (slow, and they
> eat your `max_tokens` before the answer appears).

## Manage it

```bash
examples/inference_speed/serve_best.sh --stop     # stop the server
tail -f examples/inference_speed/serve.log        # watch the server log
```

## Which profile?

| profile | flag | best for | per-stream | total throughput |
|---|---|---|---|---|
| single (default) | _none_ | one user chatting | ~6-7.7 tok/s | = per-stream |
| aggregate | `--agg` | many concurrent users | slightly lower | ~10 tok/s (12 slots) |

For interactive use (one person, one stream) pick **single** — that's the per-token
latency you feel. Use `--agg` only when serving several requests at once.

## What the config actually is

The search's best (lossless) setup, all four levers stacked:

```
model      = Qwen3.5-9B-Q4_K_M.gguf          (the 9B, cargo — unchanged)
draft      = Qwen3.5-9B-Q3_K_M.gguf          (self-speculative: same tokenizer → lossless)
draft_max  = 16                              (up to 16 draft tokens verified per pass)
KV cache   = q8_0                            (near-lossless, halves KV read bandwidth)
flash-attn = on
```

- **single:** `n_threads=6`, `n_concurrent=8`, `draft_threads=6`
- **agg:**    `n_threads=10`, `n_concurrent=12`, `draft_threads=2`

The non-obvious bit the search found: **fewer threads = faster** for single-stream,
because the 9B decode is memory-bandwidth-bound (12 threads contend on the bus; 6 get
more bandwidth each). `--agg` uses more threads + more concurrency to trade per-stream
speed for total throughput.

## Honest numbers

Measured live on port 9090 (single profile):

```
6.26 tok/s | 75 tok | def fib(n): ...           (correct Python)
6.53 tok/s | 64 tok | "The sky appears blue..."  (coherent)
6.24 tok/s | 34 tok | free -h, ps aux, top...    (correct commands)
```

~6.2-6.5 tok/s in this run. The search's peak was 7.67 single / 10.15 aggregate;
the final config re-verified at 6.19 / 7.79. tok/s on a shared VPS varies ±15-25%
(page-cache warmth, background load). The **3.6× / 5.2× improvement over the
baseline** (measured the same way) is the robust claim — not the absolute figures.

Quality is unchanged: the config is **lossless by construction** (same-model spec
draft shares the tokenizer; KV q8_0 is near-lossless; the verifier's byte-exact
greedy gate passed 61/61 attempts).