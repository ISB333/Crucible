# I Asked an AI to Make a Local Model 10× Faster. It Found 5× — and a Truth About CPU Inference.

**A verifier-grounded LLM loop, a 9-billion-parameter model on a $6/month VPS, and the night an autonomous search disproved my own prediction.**

---

I had a small model running at 2.6 tokens per second on a RAM-only VPS — no GPU, no fancy hardware, just a cheap cloud box and a lot of patience. The obvious question: *can an AI agent speed that up — without making the model dumber?*

I set an ambitious target: 30 tok/s. Not because I thought it was reachable, but because ambitious targets push systems past their comfort zone and surface what's actually possible. Then I pointed a multi-agent search at it, went to bed, and shut down my laptop.

What came back was a 3.6× speedup on single-stream generation, a 5.2× speedup on aggregate throughput, **zero quality regressions across 61 measured attempts**, and one finding I had explicitly argued was impossible. Along the way, the system caught three bugs that would have wasted the entire night — one of them diagnosed by the shepherd model, another overruled by a deterministic verifier that refused to be fooled.

This is the story, and what it taught me about why *who decides what's real* matters more than *how smart the models are*.

## The setup: a model as cargo

The architecture has three roles, and only one of them is the model being optimized:

- **The cargo:** Qwen3.5-9B, quantized to Q4_K_M (~5.5 GB), running on `llama-server` on CPU. It does nothing on its own. It's the thing being sped up.
- **The worker:** Gemini 3.1 Flash Lite, editing the inference `config` (thread counts, batch sizes, speculative-decoding draft, cache quantization). Each edit is a guess about what'll go faster.
- **The shepherd:** GLM-5.2, via Ollama Cloud, called only when the worker plateaus — to research the blocker (it has web search) and suggest a new direction.
- **The verifier:** a deterministic Python program that, for every candidate config, actually launches the 9B, streams a frozen workload, measures tokens-per-second from real token timestamps, and runs a **lossless quality gate**: the candidate's greedy output must be byte-identical to a frozen reference. Any divergence = `Fail`.

The verifier is the whole point. The thesis — borrowed from proof-systems research where verifiers decide what theorems are real — is that **the verifier, not the model's confidence, decides what's true**. The worker can be confidently wrong; the shepherd can be conservatively wrong; neither gets to declare success. Only the measurement does. And the measurement can't be gamed: the target model is integrity-checked (you can't swap in a faster, dumber model), the workload is frozen (you can't pick easy prompts), and the quality gate is byte-exact (you can't trade quality for speed and call it a win).

## Three bugs that would have wasted the night

Before the overnight run, I did what I should always do: actually check that it works. It didn't — three times.

**Bug 1: the false-positive PARTIAL.** The verifier's hole-checker scans the artifact for an unfilled-placeholder sentinel. It scanned *all* the task files — and the README documents the sentinel, in prose, to explain the mechanism. So every single artifact tripped the hole check, returned `Partial`, and the measurement never ran. The first overnight attempt produced 32 episodes of pure nothing — 32 confident-looking rounds that measured zero tokens. The shepherd (GLM-5.2) actually diagnosed this one on its own, in its first consult: *"The Partial verdict is a false positive."* I'd have caught it eventually; the shepherd caught it first.

**Bug 2: 18-minute verifies.** Each candidate config had to be measured by launching the 9B and streaming a workload. The verifier was generating 256 tokens across 14 prompts — at 2.6 tok/s, that's ~18 minutes per verify. Over a 10-hour run, that's ~30 measured configs. Not nearly enough for a search to explore. Cut to 64 tokens (still a clean rate measurement; the quality probes use a fixed 16 tokens anyway), and verifies dropped to 2-5 minutes — ~60+ measured configs overnight.

**Bug 3: the box was on fire.** Mid-investigation the load average hit 22 on 12 cores. A forgejo CI build — triggered, ironically, by my own git commits — was hammering the CPU, and available RAM had collapsed to under 5 GB while the 9B plus its speculative draft needed ~10 GB. The verify process was hitting 10,000+ major page faults, reading the model back from disk on every token. I paused the CI runner (letting the in-flight build finish, so nothing failed), load dropped, RAM recovered to 11 GB, and page faults fell to 36.

None of these were the model being slow. They were the *measurement apparatus* being broken. A confidence-grounded system would have reported "32 episodes completed, working as intended" and gone to sleep. The verifier — and the discipline of checking the measurement, not trusting the loop's self-report — caught all three.

## The night run

With the apparatus fixed, I launched the search detached (a `setsid` + `nohup` trick that reparents the process to init, so it survives the laptop shutting down — the VPS keeps it alive in its own session), and went to bed. Ten hours, four parallel workers, plateau-patience set high so it wouldn't give up early. DeepMind reportedly ran their Erdős-conjecture agent for days; I ran mine for a night, on the same principle: don't stop at the first plateau.

The numbers came back clean. 61 measured attempts. Zero partials, zero fails. Every config that scored preserved the model's exact greedy output.

| metric | baseline | best found | improvement | target |
|---|---|---|---|---|
| single-stream tok/s | 2.13 | **7.67** | 3.6× | 8.0 (within 4%) |
| aggregate tok/s | 1.96 | **10.15** | 5.2× | 30.0 (3× short) |
| quality | — | lossless (61/61) | — | — |

The best configs didn't use one trick. They stacked four: a lower-precision *draft* of the same model (speculative decoding), KV-cache quantization to q8_0, dedicated threads for the draft so it stopped stealing the target's memory bandwidth, and — here's the interesting one — *fewer* threads for the target.

## The finding I didn't expect: fewer threads = faster

This is the result that made the whole exercise worth it.

The 9B's decode is **memory-bandwidth-bound, not compute-bound**. Generating a token means reading the model's 5.5 GB of weights from RAM. The compute per token is trivial; the bottleneck is shuffling the weights through the memory bus. So the obvious tuning — "use all 12 cores" — is wrong. Twelve threads all hammer the memory bus and contend with each other; six threads each get a larger slice of bandwidth, and the bandwidth-limited decode runs *faster*.

The search found this empirically, by measurement:

```
n_threads=12:  single median 4.13 tok/s   (the default — "use all cores")
n_threads=10:  single median 5.95
n_threads= 8:  single median 6.62
n_threads= 6:  single median 6.96         ← peak (max 7.67)
n_threads= 4:  single median 6.16         (too few)
```

Cutting threads from 12 to 6 nearly **doubled** single-stream throughput, losslessly. This is the kind of counter-intuitive optimum that human tuners miss — everyone's instinct is "more cores = faster," and for a compute-bound workload it'd be right. The verifier, measuring reality instead of applying intuition, found the inverted-U and parked on its peak.

I would not have found this by hand. I'd have set `--threads 12` and moved on.

## The prediction I had to retract

Before the run, I'd done the back-of-envelope: 5.5 GB per token, ~16 GB/s memory bandwidth, so the single-stream ceiling is ~2.9 tok/s. I told the user this. The 30-target was unreachable, and even 8 (the single-stream sub-target) looked out of reach.

**I was wrong, and the search proved it.** Speculative decoding changes the arithmetic. A small *draft* model proposes several tokens; the big *target* model verifies them all in a single forward pass — one 5.5 GB weight read. When the draft is good (a same-model quant has high acceptance), several tokens get accepted per weight read. The weight read is **amortized over accepted tokens**, so effective single-stream tok/s rises *above* the no-draft ceiling.

The no-draft baseline (2.13) matched my ceiling. With a Q3-quantized draft of the same model, single-stream hit 7.67 — 3.6× the baseline, and within 4% of the 8.0 target I'd said was out of reach. The mechanism (weight-read amortization) is real and I'd simply underweighted it. The search found an improvement I had argued was impossible.

This is the honest version of the story. I'm not claiming the system was smarter than me in some general sense — I'm claiming the *measurement loop* found something my mental model had discarded, because it didn't care about my mental model. It tried the config and measured it.

## The shepherd got overruled — and that's the point

The GLM-5.2 shepherd consulted exactly once. It diagnosed a real problem — the server was timing out because loading two ~5 GB models plus a large concurrent KV cache exceeded RAM — and prescribed dropping the draft model. Reasonable advice for a crash.

The worker tried it. Six no-draft attempts. And the verifier scored the *with-draft* configs higher — because once the memory contention was cleared (Bug 3), the speculative draft's amortization paid off. The search kept the draft for 54 of 61 attempts.

The shepherd is smart. Its diagnosis of the crash was correct. Its prescription was conservative in a way that would have thrown away the single biggest lever. A system that trusts the shepherd's confidence would have dropped the draft and capped out around 4 tok/s single-stream. The system that trusts the *verifier* kept the draft and reached 7.67.

*The verifier, not the model's confidence, decides what's real.* This is the whole thesis, and it's why the architecture has three roles instead of one. The models propose; the verifier disposes. When they disagreed, the measurement won — and the measurement was right.

## What 30 tok/s would actually require

I don't want to overclaim. The aggregate target of 30 was not reached, and the search converged on the same wall I'd predicted — just at a higher point than I'd predicted. The worker's own reasoning, at convergence, read like a physicist giving up gracefully:

> *"5.5 GB/token architectural read volume is the fundamental bottleneck… n_concurrent 8-12 is the optimal batching point; beyond that memory contention overwhelms the shared-weight benefit… further gains require an architectural change (MoE)."*

Aggregate throughput scales with batching (multiple concurrent sequences share one weight read), but it flattens around n_concurrent 8-12 because the KV cache and bandwidth contention overwhelm the benefit. The dense 9B's aggregate ceiling on this box is ~10. To go further, you need to reduce the *bytes read per token* — i.e., read fewer weights — which is what a Mixture-of-Experts architecture does natively. I benchmarked one (LFM2.5-8B-A1B, 1B active parameters): ~14 single, ~20 aggregate, natively, no search. That's the architectural lever. It's not a config you can tune into a dense model.

So: 30 aggregate is beyond this VPS's memory bandwidth for a dense 9B, and the search honestly reported that ceiling instead of inflating it. It reported real ~10 aggregate and ~7.7 single, refused to fake higher numbers, and rejected every config that would have traded quality for the appearance of speed. This is the pattern I wanted to see: an honest lower bound, not a gamed score.

## What I actually learned

Three things, in order of how much they surprised me.

**1. The measurement apparatus is the product, not the models.** Three of the most important things that happened were bugs in the *measurement*, not the search. A confidence-grounded system would have reported success on a broken loop. The discipline of "actually check that it works, and trust the measurement over the loop's self-report" is what turned 32 fake episodes into 61 real ones.

**2. Verifier-grounded search finds optima that intuition misses.** The "fewer threads = faster" result is the headline. It's not deep — it's a straightforward consequence of bandwidth-bound vs compute-bound decode — but it's *counter-intuitive enough that nobody tunes it that way by default*, and the search found it in a few hours of measurement. Compound that across a dozen such levers and you get the 3.6× / 5.2× result, all lossless.

**3. The verifier overruling the shepherd is the whole game.** The smart model said "drop the draft." The measurement said "keep the draft." The measurement was right. If you're building agent systems, the question to ask isn't "how smart is my model" — it's "what decides whether the model's answer is correct." If the answer is "the model's own confidence," you will eventually be confidently wrong at scale. If the answer is "a deterministic check against reality," you get something you can trust.

I set out to test whether a verifier-grounded loop could speed up a small model. It did — 3.6× to 5.2×, losslessly. But the more interesting result is *how*: by being wrong about my prediction, overruling a smart model's conservative advice, and finding an optimum in a place no human tuner would have looked. The 9B is still slow in absolute terms. The method is the thing that got faster.

---

*The experiment is open: the harness, verifier, and search loop are in the [Crucible](https://github.com/) repo under `examples/inference_speed/`. The model is cargo; the verifier is the product. Run it overnight, shut down your laptop, and see what the measurement finds while you sleep.*