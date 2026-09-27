# Optional exercise 5 — LLM as a judge with a shared KV cache

A local model judges the flagged `p2` trace from exercise 3. For each of ten
YES/NO questions (tool use, arguments, grounding, source, units), we read the
model's probability that the **next token** is `YES` or `NO`. No text is
generated.

All ten prompts start with the same rubric and trace; only the question at the
end differs. So we can process that shared start **once**, keep its KV cache,
and reuse it for every question.

## Prepare (before the workshop)

Needs Apple Silicon. The model, `mlx-community/Qwen2.5-7B-Instruct-4bit`, is a
~4.3 GB download. The first download took about 17 minutes. It is stored in the
workshop's ignored `.cache/` folder. No Docker or API key is needed.

```bash
uv sync --extra eval --extra mlx
cd exercises/05-llm-judge
uv run python production_batch.py   # creates batch.jsonl and traces.jsonl
uv run python judge.py
```

## Files

| File | What it is |
|---|---|
| `judge.py` | The rubric, the ten questions, and the cold vs cached comparison |
| `judge_helpers.py` | Loading the trace, padding it, printing and saving the report |
| `results/` | Reports measured on an Apple M4 Max |

Read `judge.py` from top to bottom. The key parts are `yes_no_probabilities`
and the two numbered blocks in `main`: **1. cold** and **2. cached**.

## Two kinds of percentages

- `YES %` and `NO %`: probability over the model's whole vocabulary. `other %`
  is everything else (for example "Yes" or "The").
- `YES vs NO %`: only the two labels compared, so it sums to 100% with NO.

These are the model's preferences under this prompt, **not calibrated
probabilities that the judgment is correct**. The expected answers in
`QUESTIONS` were written by hand for the unfixed `p2` trace. If you change the
agent or the trace, update them.

## Measured on the workshop Mac

[`results/judge_p2.json`](results/judge_p2.json) (the real trace) and
[`results/judge_p2_long_trace.json`](results/judge_p2_long_trace.json)
(`--synthetic-spans 32`: the same facts padded with made-up, trace-shaped
context). Each question adds 14–21 tokens.

| Ten judgments | p2 trace, 847 shared tokens | long trace, 8,002 shared tokens |
|---|---:|---:|
| 1. Cold: full prompt every time | 10.25 s | 116.83 s |
| 2a. Prefill the shared prefix once | 0.97 s | 10.13 s |
| 2b. Ten question tails from the cache | 0.46 s | 0.70 s |
| First batch including prefill | 1.44 s (7.1×) | 10.83 s (10.8×) |
| Once the cache is warm | 22.1× | 168.0× |

Both paths chose the expected label for **10/10** questions on both traces.
On the p2 trace, cold and cached `YES vs NO %` matched within 0.0004
percentage points. On the long trace they matched within 0.0001 points except
for the least certain question, `temperature_14c`: 84.80% cached vs 85.59% cold.
That gap comes from low-precision arithmetic over 8k tokens, not from the cache
holding different content. Model loading is not timed. These are local
measurements, not general claims.

## What to inspect

1. Compare the shared prefix with each question's `tail_tokens`. The rubric and
   trace are shared; the question is the changing suffix.
2. Compare the cold and cached times. Why is the long trace's speedup so much
   bigger?
3. Compare `YES %` with `YES vs NO %`, and `yes_vs_no_pct` with
   `cold_yes_vs_no_pct` in the report.
4. Change one detail in the trace, run `production_batch.py` again, and rerun.
   The old cache is useless now because the token prefix changed. A different
   rubric, tokenizer or model also needs a new cache.

Each question gets a **copy** of the prefix cache (`copy.deepcopy`). If you
reused one cache, earlier questions would leak into later judgments.

References: the MLX LM [prompt cache example](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/examples/chat.py),
[cache prompt command](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/cache_prompt.py),
and [unified memory on Apple Silicon](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).
