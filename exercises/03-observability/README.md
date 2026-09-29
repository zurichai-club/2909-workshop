# Exercise 3 — find a failure in traces

## Run

```bash
docker compose up -d                  # from the workshop folder, if not running yet
cd exercises/03-observability
uv run --env-file ../../.env python production_batch.py
```

Langfuse: <http://127.0.0.1:3000> (`workshop@example.com` / `workshop`).

## Task

1. Find the traces the script printed in Langfuse.
2. Find out what went wrong in the flagged `p2` run.
