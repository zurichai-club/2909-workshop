# Optional exercise 5 — LLM as a judge with a shared KV cache

Checkpoint: `git checkout ex5-start` (create your own branch before editing).

This builds on Exercise 3's flagged `p2` trace. It uses a local MLX model to
answer ten yes/no questions about the same trace. The questions cover tool use,
arguments, grounding, source, and whether the final answer met the user's
requested units. Expected labels are in `scripts/judge_cache_benchmark.py`.
Run this on `main`, where `p2` intentionally still has the Fahrenheit failure.
If you change the agent or trace, update the expected labels before using the
accuracy number as a quality check.

## Prepare

Apple Silicon with a working Metal device is required. The default model is
`mlx-community/Qwen2.5-7B-Instruct-4bit`, a roughly 4.3 GB download. The model
stays in this workshop's ignored `.cache/` directory. Docker and model API
keys are not required for this exercise.

```bash
uv sync --extra eval --extra mlx
uv run python scripts/production_batch.py
uv run python scripts/judge_cache_benchmark.py
```

The report is written to `results/judge_kv_cache.json`. It excludes model load
and tokenization from the timing, but includes the cached path's one-time
prefill and per-question cache copies. Two rounds reverse the run order.

## Measured on the workshop Mac

The saved [benchmark report](../results/judge_kv_cache.json) used the 4-bit
Qwen 2.5 7B model on an Apple M4 Max. The ten prompts shared **860 tokens**;
each question added **14–21 tokens**. After a warm-up, the median of two rounds
was:

| Ten judgments | Time |
|---|---:|
| Reprocess the full trace each time | 11.06 s |
| Prefill once, then copy the cache for each question | 2.35 s |

That is **4.71× faster** for this trace and machine. The cached time includes
about **1.04 s** of shared prefill; median per-question time including the copy
was **0.13 s** versus **1.14 s** cold. Both paths gave identical answers and
matched all **10/10** hand-labeled judgments. Model loading and the first model
download are outside the timed comparison. Peak process memory was about
**5.0 GB**. These figures are a local measurement, not a general model claim.
As a quality check, a [0.5B smoke run](../results/judge_kv_cache_0p5b_smoke.json)
was faster but matched only **5/10** labels; all ten responses were "yes"-leaning.
The judge needs calibration against human labels even when the cache works.

## What to inspect

1. Compare `shared_prefix_tokens` with `tail_tokens`. The rubric and trace
   should be shared; the question and answer header are the changing suffix.
2. Compare cold and cached total times, then inspect the per-question times.
   The first cached run pays for prefill; later questions should avoid it.
3. Compare `cold_correct`, `cached_correct`, and `labels_identical`. Faster
   judgments are useful only if their answers remain acceptable.
4. Edit one trace detail, regenerate the batch, and rerun. The old KV cache is
   no longer valid because the token prefix has changed. A different rubric,
   tokenizer, or model likewise requires a new cache.

The script computes the longest *identical token* prefix across the ten full
chat prompts. It then precomputes that prefix once and copies the resulting
cache before adding each question. Reusing one mutable cache sequentially
would mix previous questions and answers into later judgments. This technique
reduces repeated prefill work; it does not remove decoding time, and its value
depends on prefix length, model, hardware, and cache-copy cost.

MLX LM documents the cache pattern in its [prompt cache example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py)
and [cache prompt command](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/cache_prompt.py).
The default model is the [MLX Community Qwen 2.5 7B 4-bit release](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit).
