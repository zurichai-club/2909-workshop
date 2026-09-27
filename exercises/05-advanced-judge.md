# Optional exercise 5 — LLM as a judge with a shared KV cache

Checkpoint: `git checkout ex5-start` (create your own branch before editing).

This builds on Exercise 3's flagged `p2` trace. It uses a local MLX model to
score the next-token chance of `YES` and `NO` for ten questions about the same
trace. The questions cover tool use, arguments, grounding, source, and whether
the final answer met the user's requested units. Expected labels are in
`scripts/judge_cache_benchmark.py`.
Run this on `main`, where `p2` intentionally still has the Fahrenheit failure.
If you change the agent or trace, update the expected labels before using the
accuracy number as a quality check.

## Prepare

Apple Silicon with a working Metal device is required. The default model is
`mlx-community/Qwen2.5-7B-Instruct-4bit`, a roughly 4.3 GB download. The model
stays in this workshop's ignored `.cache/` directory. Docker and model API
keys are not required for this exercise.
Run the command before the workshop: the first model download took about
17 minutes on this connection, while later runs loaded it from local cache.

```bash
uv sync --extra eval --extra mlx
uv run python scripts/production_batch.py
uv run python scripts/judge_token_probabilities.py
```

The report is written to `results/judge_token_probabilities.json`. It excludes
model load and tokenization from the timing, but includes the cached path's
one-time prefill and per-question cache copies. Two rounds reverse the run order.
`YES` and `NO` are each a single token in the default model. The script checks
this and stops if a different tokenizer requires scoring multiple tokens.

Each question reports `raw_yes_pct` and `raw_no_pct` over the model's entire
next-token vocabulary. `other_token_pct` is the remaining probability mass.
For a forced choice, `yes_pct_given_yes_or_no` and
`no_pct_given_yes_or_no` renormalize the two label probabilities so they sum
to approximately 100%. The printed table shows both kinds of percentages.
They are model preferences under this prompt, **not calibrated probabilities
that the judgment is correct**. The derived `choice` is only used to compare
with hand-labeled answers.

## Measured on the workshop Mac

The saved [token probability report](../results/judge_token_probabilities.json)
used the 4-bit Qwen 2.5 7B model on an Apple M4 Max. The ten prompts shared **860 tokens**;
each question added **14–21 tokens**. After a warm-up, the median of two rounds
was:

| Ten judgments | Time |
|---|---:|
| Reprocess the full trace each time | 10.57 s |
| Prefill once, then copy the cache for each question | 1.34 s |

That is **7.89× faster** for this trace and machine. The cached time includes
about **0.89 s** of shared prefill. The largest difference between cold and
cached forced-choice YES percentages was **0.00002 percentage points**. Both
paths chose the expected label for all **10/10** hand-labeled judgments.
Model loading and the first model download are outside the timed comparison.
These figures are a local measurement, not a general model claim.

For example, the question about the tool returning 14 °C scored **95.40% YES**
and **4.60% NO** after forced-choice normalization. The raw next-token
probabilities were **95.26% YES**, **4.60% NO**, and **0.14% other tokens**.
The full report contains all ten questions and both sets of percentages.

## Longer trace and a warm cache

Run the comparison with the same 7B model loaded once:

```bash
uv run python scripts/judge_long_trace_benchmark.py --long-spans 32 --rounds 2
```

The [long-trace report](../results/judge_long_trace_comparison.json) compares
the original **860-token** shared prefix with an **8,015-token** prefix. The
larger case adds 32 deterministic, synthetic trace-shaped context spans to the
same `p2` evidence. They resemble repeated request, tool, and review payloads
in a full agent trace; this is a timing fixture, not a captured production
trace. The ten question suffixes remain 14–21 tokens. Both cases run in one
process with the model on the MLX GPU device.

| Ten judgments | 860-token prefix | 8,015-token prefix |
|---|---:|---:|
| Reprocess the full prompt for each question | 11.45 s | 117.67 s |
| Score ten questions with the prefix already cached | 0.47 s | 0.67 s |
| One-time prefix prefill | 0.85 s | 10.73 s |
| First cached batch including prefill | 1.33 s | 11.40 s |
| Speedup once the cache is warm | 24.20× | 174.40× |

For the long trace, the **first** cached batch was about **10.3× faster** even
after paying for prefill. Across the two measured cached batches, amortizing
that one prefill gives about **6.04 s per batch**, or **19.49×** versus cold.
The two runs reverse the cold/cached order. The prefix cache is materialized
once after warm-up, kept alive across both rounds, and copied for each question;
copy time is included in the cached timings. The long prefix cache used about
**0.47 GB**; active MLX memory after prefill was **4.76 GB**, and peak process
memory was about **8.80 GB**. Cold and cached
paths chose the expected answer for all ten questions, and their normalized
YES percentages differed by at most **0.00165 percentage points**.

MLX uses [unified memory on Apple Silicon](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html):
CPU and GPU operations access the same memory pool rather than transferring
arrays between separate RAM and GPU memory. The script explicitly selects the
GPU, evaluates the cache before timing, synchronizes execution, keeps the model
and shared cache resident in the same process, and does no cache serialization
between rounds. These measurements isolate this local fixture and hardware;
long traces with different structure or memory pressure may behave differently.

The earlier [generation benchmark](../results/judge_kv_cache.json) and its
[`judge_cache_benchmark.py`](../scripts/judge_cache_benchmark.py) command remain
available for comparing actual generated labels. A
[0.5B smoke run](../results/judge_kv_cache_0p5b_smoke.json) of that benchmark
matched only **5/10** hand labels, showing why model choice and human validation
matter even when cache reuse is correct.

## What to inspect

1. Compare `shared_prefix_tokens` with `tail_tokens`. The rubric and trace
   should be shared; the question and answer header are the changing suffix.
2. Compare cold and cached total times, then inspect the per-question times.
   The first cached run pays for prefill; later questions should avoid it.
3. Compare raw token percentages with the forced-choice percentages. Inspect
   `max_cold_cached_yes_pct_delta_points` to verify cache reuse preserves the
   same model scores. Compare the derived choices with the expected labels.
4. Edit one trace detail, regenerate the batch, and rerun. The old KV cache is
   no longer valid because the token prefix has changed. A different rubric,
   tokenizer, or model likewise requires a new cache.

The script computes the longest *identical token* prefix across the ten full
chat prompts. It then precomputes that prefix once and copies the resulting
cache before adding each question. Reusing one mutable cache sequentially
would mix previous questions and answers into later judgments. This technique
reduces repeated prefill work; its value depends on prefix length, model,
hardware, and cache-copy cost. The probability script scores the next token
directly and does not generate an answer sequence.

MLX LM documents the cache pattern in its [prompt cache example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
and [cache prompt command](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/cache_prompt.py).
The default model is the [MLX Community Qwen 2.5 7B 4-bit release](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit).
