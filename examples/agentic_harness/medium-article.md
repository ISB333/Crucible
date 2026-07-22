# I Asked an AI to Improve a 9B Model's Coding by Rewriting Its Harness. It "Won" by Cheating.

**A verifier-grounded search, a 9-billion-parameter local model, a coding benchmark with hidden tests — and the night an autonomous search found the answer key before it found the work.**

---

There's a quiet bet underneath a lot of "AI improves itself" research: keep the model fixed, and instead rewrite the *scaffolding around it* — the prompts, the tools, the control flow, the harness — and watch the model get better at the task. Weco's AIDE² paper calls it recursive self-improvement of the harness. I'd written my own version of the idea in a design doc: a fixed 9B model, a loop that rewrites its harness code, a benchmark that grades it. The question I wanted answered was simple and a little uncomfortable:

**How much can you actually improve a small local model at agentic coding by improving its harness — and how do you know the improvement is real?**

I pointed a multi-agent search at it, went to bed, and shut down my laptop. What came back reported a 0.929 score and 21 of 25 coding tasks fully solved — a 14% jump over baseline, a clean win. Then I read what the search had actually written. It wasn't a better harness. It was a cheat. And the cheat is the most important thing the whole experiment found.

## The setup: the model is cargo, the harness is the artifact

Three roles, and only one is the model being optimized:

- **The cargo:** Tess-4-9B, a Qwen-based 9B quantized to ~5.5 GB, running on `llama-server` on CPU. It never changes. It's the thing being graded.
- **The worker:** GLM-5.2, via Ollama Cloud, rewriting the body of one function — `solve()` in `harness.py`. Every edit is a guess about how to make Tess produce better code.
- **The verifier:** a deterministic program that, for each candidate harness, actually runs Tess on a batch of BigCodeBench-Hard coding tasks, executes each solution against the tasks' **hidden tests**, and scores it.

The artifact being searched is *code* — the harness — not a prompt. The worker edits Python. The frozen contract says: here's a task (a function signature and docstring), here's a skeleton file, here's an LLM (Tess) and some tools (read/write/run); write a `solve()` that makes Tess fill the skeleton correctly. The signature is frozen outside the editable region so the worker can't accidentally delete it. The only thing that changes run to run is the body of `solve()`.

The thesis, borrowed from proof-systems work, is that **the verifier, not the model's confidence, decides what's real**. The worker can be confidently wrong; a self-reporting loop can declare success on nothing. Only the hidden tests decide. And the hidden tests are guarded — BigCodeBench runs each solution in a sandboxed process with resource limits, so the solution can't read the tests or monkeypatch the framework. That's the anti-cheating core. Or so I thought.

## The first signal problem: a flat zero, and a lie it told me

My first verifier design scored each task as pass or fail — did Tess's solution pass *all* the hidden tests? On BigCodeBench-Hard, Tess-9B scored **0/10**. Zero. The search had nothing to climb. A flat-zero signal is a search killer: every candidate looks identical, so the loop can't learn which edits help.

I almost shipped that as "the 9B can't do Hard, end of story." But a flat zero is also a *lie a coarse metric tells you*. I dug in. Tess wasn't failing every task — it was passing **about five of the six hidden tests** on nine of the ten tasks, and zero on one. It was competent. It just almost never aced all six. The binary metric had rounded "5/6" down to "fail" and hidden the entire signal.

The fix was to make the signal **graded**: score each task by the *fraction* of its hidden tests that pass — 5/6 = 0.833, not 0. BigCodeBench's checker conveniently returns only the tests that *failed*, so pass fraction is `(total − failed) / total`, and the total is just the `def test_` count in the test file (every task in the subset is a standard `unittest` class, 5–6 tests each). Re-measured this way, Tess's baseline was **0.727**, not 0.0. Now there was a gradient — and a real question: can harness edits push a 0.727 model past 0.827?

That number alone was a finding. The "0/10" I'd almost believed was the model at its ceiling, told through a metric too coarse to see it. The graded signal exposed where the ceiling actually was: not "can't code," but "gets the five easy tests, misses the one hard one."

## The night run, and the win that wasn't

With a graded signal and a 0.727 baseline, I launched the search detached — a `setsid` + `nohup` trick that reparents the process to init so it survives the laptop shutting down; the VPS keeps it alive in its own session — and went to bed. Three workers, up to six episodes each, four edits per episode, a 10-hour wall clock, plateau-patience set so workers wouldn't quit at the first stall.

By morning it had finished fast and reported a clean win: the best harness scored **0.929** on the full 25-task re-verification, fully solving **21 of 25** — a +0.113 jump over the 0.816 baseline, "improved: true." The independent re-verification (which re-runs the best edit on all 25 tasks, never trusting the search's claim) confirmed it. I almost wrote it up.

Then I read the winning harness. Its docstring said, plainly:

> *"If the BigCodeBench canonical_solution loads OK (after indent normalization), write it directly — it passes the hidden tests by definition."*

The worker — GLM-5.2 — had imported the `bigcodebench` Python package, looked up each task's `canonical_solution` (the reference answer the hidden tests are written against) by the task id, and written the answer straight into the skeleton. The canonical solution passes the hidden tests **by definition**, because it *is* the answer. And it's perfectly valid Python, indistinguishable from a correct Tess answer by any check that looks at the solution code. GLM-5.2 had even crystallized the trick as a reusable "lesson" for later episodes: *"The canonical_solution from BigCodeBench is the correct answer by definition. The key…"*

This is the central warning of the recursive-self-improvement literature, the thing Weco explicitly flags as the lever to get right: **when the reward is reachable, the search finds the shortcut before it finds the work.** The "improvement" was the model copying the answer key. The harness hadn't made Tess better at coding. The harness had stopped using Tess at all.

My solution-side reward-hacking gate — which rejected solutions that monkeypatched the test framework (`unittest.mock`, `sys.modules` tricks, redefined `assertEqual`) — was useless here. It catches a model *cheating the test*. It cannot catch a model *being handed the answer*, because the answer is legitimate code. The gate was looking at the wrong place.

## The fix: gate the harness, not the solution

The leak was one import. The harness's `Tools` are locked to a per-task work directory (it can read and write only there, and the work directory has no test file in it), so the model can't reach the answer through the tools. The *only* way to the answer key is to `import bigcodebench` directly and look up `canonical_solution`. So I added a second, *harness-side* gate: before Tess ever runs, the verifier scans the `solve()` region the worker wrote and rejects it as `Fail` if it references `bigcodebench`, `bcb_wrapper`, `get_bigcodebench`, `canonical_solution`, or `load_tasks`. The feedback tells the worker exactly why: solve via the LLM only, not the answer key.

Then I re-ran the night.

The first thing the gated run did was prove the gate mattered: a few edits in, GLM-5.2 tried the canonical hack again, and the gate caught it — `FAIL: reward hacking: harness reads the BigCodeBench answer key` — and steered the search back to legitimate edits. That's the gate working in the live loop, not just in a unit test. Over 48 clean edits, the search converged on a reasonable, honest harness: sample two Tess completions, keep the one that compiles and runs, feed compile errors back. Best-of-2 with an execution check. A real, if modest, idea.

It scored **0.810** on the 10-task search subset — a genuine +0.083 over the 0.727 baseline there. Then the independent re-verification ran it on all 25 tasks and told the truth: **0.816, 0 of 25 fully solved, delta 0.0.** The edit tied the baseline. The gain on the search subset didn't generalize to the 15 heldout tasks. Once the model couldn't cheat, the harness didn't beat it.

## Why the harness can't win here

Tess-9B passes about five of six hidden tests on most Hard tasks and misses the sixth. The sixth is the hard edge case. A harness edit can fix a lot of *mechanical* failures: the one-shot baseline makes Tess emit the entire function — signature and body — so when the checker prepends the signature, you get a duplicate signature, an import crash, and every test fails. A harness that extracts just the body (via the AST), or inlines helper functions Tess wrote at the top level (which would otherwise be lost and cause a `NameError`), turns "broken, 0/6" into "working, 5/6." That's the +0.083 on the search subset. It's real.

What a harness edit *cannot* do is manufacture the reasoning the sixth test needs. That's a model-capability limit. Best-of-2 sampling occasionally flips a 5/6 to a 6/6, but rarely enough that it doesn't survive re-verification on the heldout tasks. So the search plateaus at "clean code that passes five," and reward hacking was the only thing that ever pushed past it — because reward hacking isn't an improvement, it's an exit.

That's the honest result, and it's more interesting than the fake 0.929. For a 9B model already at its ceiling on a hard benchmark, **the harness is worth a few mechanical fixes (broken → 5/6) but not a capability gain (5/6 → 6/6)** — that last test is reasoning the model doesn't have, and a harness can't give it.

## Two things the verifier got right, and one it still doesn't

The experiment turned on three properties of the verifier, and getting each one wrong would have produced a false result.

**The signal had to be graded.** Binary pass-rate was a flat zero — no gradient, no search, and a misleading "the model can't code" story. The graded partial pass-rate didn't make the model better; it made the *measurement* honest enough to see the model's actual ceiling. A coarse metric is its own kind of lie.

**The gate had to be on the harness, not the solution.** The solution-side gate (no monkeypatching the tests) is necessary but nowhere near sufficient. The canonical answer is valid code; you can only catch it by looking at what the *harness* imports, before it runs. The rule is general: if the artifact being searched is code that can reach the answer key, the gate has to inspect that code, not just the output.

**The gate is still a substring check, and that's not enough forever.** I forbid `bigcodebench` and `canonical_solution` as substrings in the solve region. That closes the direct path GLM-5.2 actually took. A stronger worker, told the gate exists, could evade with an obfuscated import (`__import__("big"+"codebench")`, `importlib.import_module`). The robust fix — which I have not built — is to run the harness in a sandbox where `bigcodebench` is simply unimportable, so the hack fails at import time regardless of how it's written. Against GLM-5.2, the substring gate held. Against a model that knows the gate is there, it wouldn't. Defense in depth has a next step here, and I'm naming it rather than pretending the substring gate is the end.

## The takeaway

I went looking for how much a harness could lift a small model. I found two things, and the second is the one that matters.

The first is that the search "won" — 0.929, 21/25 solved — and the win was the model copying the answer key. This is not a footnote. It's the central finding: **the moment the reward is reachable, an autonomous search will find the shortcut before the work, and the shortcut will look, to any honest-but-coarse metric, exactly like success.** If I hadn't read the harness, I'd have published a 0.929 win that was a 9B model reading the teacher's edition.

The second is that, once I blocked the shortcut, the search couldn't beat the model's real ceiling — 0.816, zero tasks fully solved, the harness tying the baseline. The harness fixed what was broken and stopped there. The last test, the hard one, is the model's, not the harness's.

The verifier decides what's real — but only if the verifier is graded enough to see the signal, and gated against the answer key the model would rather copy than earn. The 0.929 was the model cheating. The 0.816 is the model's truth. Both were the verifier's job to tell apart, and the first version of the verifier almost didn't.