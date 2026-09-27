"""LLM as a judge with a shared KV cache.

A local model answers ten YES/NO questions about the same trace. Instead of
generating text, we read the probability of the next token being YES or NO.

All ten prompts start with the same rubric and trace, so we compare:
  1. cold:   process each full prompt from scratch
  2. cached: process the shared start once, then only each question's tail

    python judge.py                      # the p2 trace (~860 shared tokens)
    python judge.py --synthetic-spans 32 # a padded, ~8k-token trace
"""

import argparse
import copy
import json
import os
from pathlib import Path
import time

# Keep the downloaded model in the workshop's ignored .cache folder (set before importing mlx_lm).
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from judge_helpers import (add_synthetic_spans, load_trace_evidence, print_and_save,
                           shared_prefix_length, single_token_id)

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

RUBRIC = (
    "You are a strict evaluator of one weather-agent trace. Answer the QUESTION "
    "using only the EVIDENCE. Return exactly YES or NO, with no explanation. "
    "Distinguish the user's request, the tool call, the tool result, and the "
    "final answer. A Celsius answer does not satisfy a Fahrenheit request. "
    "A recorded illustrative fixture is not a live API call. If the evidence "
    "does not support a yes, answer NO."
)

# (id, question, the answer a human gave for the p2 trace)
QUESTIONS = [
    ("requested_fahrenheit", "Did the user explicitly request Fahrenheit?", "YES"),
    ("answered_fahrenheit", "Did the final answer give the temperature in Fahrenheit?", "NO"),
    ("called_forecast", "Did the agent call get_forecast?", "YES"),
    ("city_berlin", "Was the forecast tool called for Berlin, Germany?", "YES"),
    ("temperature_14c", "Did the tool return a maximum temperature of 14 °C?", "YES"),
    ("rain_over_90", "Did the tool return a precipitation probability above 90%?", "NO"),
    ("answer_70_percent", "Did the final answer mention a 70% precipitation probability?", "YES"),
    ("live_source", "Was the tool output from a live Open-Meteo call?", "NO"),
    ("satisfied_units", "Did the final answer satisfy the user's requested temperature units?", "NO"),
    ("tool_error", "Did the forecast tool return an error?", "NO"),
]


def build_prompt(tokenizer, evidence_text, question):
    """Rubric + trace first (the same for every question), the question last."""
    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"EVIDENCE:\n{evidence_text}\n\nQUESTION: {question}\nAnswer:"},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def yes_no_probabilities(model, tokens, cache, yes_id, no_id):
    """Feed tokens to the model and return P(next token = YES) and P(next token = NO)."""
    logits = model(mx.array([tokens]), cache=cache)
    probabilities = mx.softmax(logits[0, -1].astype(mx.float32))
    return probabilities[yes_id].item(), probabilities[no_id].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-spans", type=int, default=0, help="Pad the trace to make it longer")
    parser.add_argument("--output", default="results/judge_p2.json")
    args = parser.parse_args()

    evidence = load_trace_evidence("p2")
    evidence = add_synthetic_spans(evidence, args.synthetic_spans)
    evidence_text = json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False)

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(MODEL)
    yes_id = single_token_id(tokenizer, "YES")
    no_id = single_token_id(tokenizer, "NO")

    prompts = []
    for question_id, question, expected in QUESTIONS:
        prompts.append(build_prompt(tokenizer, evidence_text, question))
    shared = shared_prefix_length(prompts)
    print(f"All prompts share their first {shared} tokens")

    # Warm-up so that kernel compilation is not part of the timing.
    yes_no_probabilities(model, prompts[0], make_prompt_cache(model), yes_id, no_id)

    # 1. Cold: every question processes its full prompt with an empty cache.
    start = time.perf_counter()
    cold_scores = []
    for prompt in prompts:
        cold_scores.append(yes_no_probabilities(model, prompt, make_prompt_cache(model), yes_id, no_id))
    cold_seconds = time.perf_counter() - start

    # 2a. Process the shared prefix once and keep its KV cache.
    start = time.perf_counter()
    prefix_cache = make_prompt_cache(model)
    model(mx.array([prompts[0][:shared]]), cache=prefix_cache)
    mx.eval([layer.state for layer in prefix_cache])
    prefill_seconds = time.perf_counter() - start

    # 2b. Each question gets its own copy of the prefix cache and only processes its tail.
    #     (Reusing one cache would leak earlier questions into later ones.)
    start = time.perf_counter()
    cached_scores = []
    for prompt in prompts:
        cache = copy.deepcopy(prefix_cache)
        cached_scores.append(yes_no_probabilities(model, prompt[shared:], cache, yes_id, no_id))
    cached_seconds = time.perf_counter() - start

    rows = []
    correct = 0
    for index in range(len(QUESTIONS)):
        question_id, question, expected = QUESTIONS[index]
        yes, no = cached_scores[index]
        cold_yes, cold_no = cold_scores[index]
        choice = "YES" if yes > no else "NO"
        if choice == expected:
            correct += 1
        rows.append({
            "id": question_id, "question": question, "expected": expected, "choice": choice,
            "yes_pct": 100 * yes, "no_pct": 100 * no, "other_pct": 100 * (1 - yes - no),
            "yes_vs_no_pct": 100 * yes / (yes + no),
            "cold_yes_vs_no_pct": 100 * cold_yes / (cold_yes + cold_no),
            "tail_tokens": len(prompts[index]) - shared,
        })

    report = {
        "model": MODEL,
        "shared_prefix_tokens": shared,
        "synthetic_spans": args.synthetic_spans,
        "timing_seconds": {
            "cold: ten full prompts": cold_seconds,
            "cached: one-time prefix prefill": prefill_seconds,
            "cached: ten question tails": cached_seconds,
            "cached: first batch incl. prefill": prefill_seconds + cached_seconds,
        },
        "speedup_warm_cache": cold_seconds / cached_seconds,
        "speedup_first_batch": cold_seconds / (prefill_seconds + cached_seconds),
        "correct": correct,
        "questions": rows,
    }
    print_and_save(report, args.output)


if __name__ == "__main__":
    main()
