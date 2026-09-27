# Exercise 3 — find a failure in traces

Every agent run creates a trace: one span for the run, one per model step and
one for the tool call. Here you look at "production" traces in Langfuse.

```bash
docker compose up -d                  # from the workshop folder, if not running yet
cd exercises/03-observability
uv run --env-file ../../.env python production_batch.py
```

## Files

| File | What it is |
|---|---|
| `production_batch.py` | Runs four messy questions and writes `batch.jsonl` (query ID, flag, trace ID) |
| `weather_agent/agent.py` | Where the spans are created (`start_as_current_span`) |
| `weather_agent/tracing.py` | OpenTelemetry setup; sends spans to `traces.jsonl` and Langfuse |

## Task

1. Open <http://127.0.0.1:3000>, go to **Traces**, and search for the trace IDs
   the script printed.
2. Each trace shows the agent run, the model plan, the tool call (when needed)
   and the model answer. Compare the **input** and **output** of each span.
3. Find the flagged `p2` run. What did the user ask for, and which units are in
   the final answer? Check the tool output: was the fault in the data, or in
   how the answer was written?

The same data is in `traces.jsonl` and `batch.jsonl` if the UI is not available.
You can also trace a single question: `uv run --env-file ../../.env python ask.py "Will it rain in Zurich tomorrow?"`.

<details>
<summary>Solution</summary>

The user asked for Fahrenheit. The tool correctly returned 14 °C, but the final
answer still says °C. The data was fine; answer generation (`mock_answer`)
ignored the request. You fix it in exercise 4.

</details>
