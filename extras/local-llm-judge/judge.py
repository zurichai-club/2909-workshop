"""A local LLM judge: YES/NO questions about one trace, answered by a model on your Mac.

We don't let the model write text. We read its probability that the next token is YES.
All questions share the same long start (rubric + trace), so we also compare:
  1. without a cache: the model reads every full prompt from scratch
  2. with MLX's prompt cache: it reuses the shared start and reads only each question

    python production_batch.py   # creates the p2 trace
    python judge.py
"""

import json
import os
from pathlib import Path
import time

# Keep the downloaded model in the workshop's ignored .cache folder (set before importing mlx_lm).
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

RUBRIC = (
    "You are a strict evaluator of one weather-agent trace. Answer the QUESTION "
    "using only the EVIDENCE. Return exactly YES or NO, with no explanation. "
    "Distinguish the user's request, the tool call, the tool result, and the "
    "final answer. A Celsius answer does not satisfy a Fahrenheit request. "
    "A recorded illustrative fixture is not a live API call. If the evidence "
    "does not support a yes, answer NO."
)

# (question, the answer a human gave for the p2 trace)
QUESTIONS = [
    ("Did the user explicitly request Fahrenheit?", "YES"),
    ("Did the final answer give the temperature in Fahrenheit?", "NO"),
    ("Did the agent call get_forecast?", "YES"),
    ("Was the forecast tool called for Berlin, Germany?", "YES"),
    ("Did the tool return a maximum temperature of 14 °C?", "YES"),
    ("Did the tool return a precipitation probability above 90%?", "NO"),
    ("Did the final answer mention a 70% precipitation probability?", "YES"),
    ("Was the tool output from a live Open-Meteo call?", "NO"),
    ("Did the final answer satisfy the user's requested temperature units?", "NO"),
    ("Did the forecast tool return an error?", "NO"),
]


def find_run(query_id):
    """The row of one production run in batch.jsonl."""
    for line in open("batch.jsonl"):
        row = json.loads(line)
        if row["query_id"] == query_id:
            return row


def load_trace(query_id):
    """What the judge sees for one production run: question, tool calls, answer and the spans."""
    row = find_run(query_id)
    spans = []
    for line in open("traces.jsonl"):
        span = json.loads(line)
        if span["trace_id"] == row["trace_id"]:
            spans.append({"name": span["name"],
                          "input": span["attributes"].get("langfuse.observation.input"),
                          "output": span["attributes"].get("langfuse.observation.output")})
    return {"user_question": row["input"], "tool_calls": row["tool_calls"],
            "final_answer": row["answer"], "trace_spans": spans}


def build_prompt(tokenizer, trace, question):
    """Rubric + trace first (the same for every question), the question last. Returns token ids."""
    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"EVIDENCE:\n{json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False)}\n\nQUESTION: {question}\nAnswer:"},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def probability_of_yes(tokens, cache):
    """Feed tokens to the model and compare the next-token probabilities of YES and NO."""
    logits = model(mx.array([tokens]), cache=cache)
    probabilities = mx.softmax(logits[0, -1].astype(mx.float32))
    yes = probabilities[YES].item()
    no = probabilities[NO].item()
    return yes / (yes + no)


# ----------------------------------------------------------------------------
# Load the model and build one prompt per question.
model, tokenizer = load(MODEL)
YES = tokenizer.encode("YES", add_special_tokens=False)[0]  # YES and NO are single tokens for this model
NO = tokenizer.encode("NO", add_special_tokens=False)[0]

trace = load_trace("p2")
prompts = []
for question, expected in QUESTIONS:
    prompts.append(build_prompt(tokenizer, trace, question))

probability_of_yes(prompts[0], make_prompt_cache(model))  # warm-up, so the first timing is fair

# ----------------------------------------------------------------------------
# 1. Without a cache: every question reads its full prompt.
start = time.perf_counter()
for prompt in prompts:
    probability_of_yes(prompt, make_prompt_cache(model))
seconds_without_cache = time.perf_counter() - start

# ----------------------------------------------------------------------------
# 2. With MLX's prompt cache. It stores the cache of each prompt it has seen. For a new prompt it
#    finds the most similar one, trims its cache back to the shared start, and returns a copy of
#    that cache plus the tokens still to read: here, only the question.
start = time.perf_counter()
prompt_cache = LRUPromptCache()
scores = []
for prompt in prompts:
    cache, rest = prompt_cache.fetch_nearest_cache(MODEL, prompt)
    if cache is None:  # the first prompt: nothing cached yet
        cache = make_prompt_cache(model)
    scores.append(probability_of_yes(rest, cache))
    prompt_cache.insert_cache(MODEL, prompt, cache)
    print(f"read {len(rest)} of {len(prompt)} tokens")
seconds_with_cache = time.perf_counter() - start

# ----------------------------------------------------------------------------
print(f"\n{'P(YES)':>7}  {'judge':<5} {'human':<5}  question")
correct = 0
for (question, expected), score in zip(QUESTIONS, scores):
    verdict = "YES" if score > 0.5 else "NO"
    correct += verdict == expected
    print(f"{score:7.1%}  {verdict:<5} {expected:<5}  {question}")
print(f"\nThe judge agreed with the human on {correct}/{len(QUESTIONS)} questions.")
print(f"Without a cache: {seconds_without_cache:.2f} s. With a cache: {seconds_with_cache:.2f} s "
      f"({seconds_without_cache / seconds_with_cache:.1f}x faster).")
