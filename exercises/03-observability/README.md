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

<details>
<summary>Solution</summary>

The user asked for Fahrenheit. The tool correctly returned 14 °C, but the final
answer still says °C. The data was fine; answer generation (`mock_answer`)
ignored the request. You fix it in exercise 4.

</details>
