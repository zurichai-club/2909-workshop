# Exercise 3 — find a failure in traces

Checkpoint: `git checkout ex3-start`.

Start Langfuse with `docker compose up -d`, then run
`uv run --env-file .env python scripts/production_batch.py`.
Open <http://127.0.0.1:3000>, select **Traces**, and use the trace IDs printed
by the script. Each trace should show the agent run, model plan, tool call
(when needed), and model answer. Compare `langfuse.observation.input` and
`langfuse.observation.output` on these spans. The same data is in
`traces.jsonl` for local inspection.

Find the flagged `p2` run. What did the user request, and what units appeared
in the final answer? Look at the tool output to decide whether the fault was
in the data or in answer generation. The `batch.jsonl` file gives the same
query ID, flag and trace ID when the UI is unavailable.
