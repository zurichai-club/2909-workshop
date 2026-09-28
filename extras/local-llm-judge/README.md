# Run
```bash
uv sync --extra eval --extra mlx
cd extras/local-llm-judge
uv run python production_batch.py   # creates batch.jsonl and traces.jsonl
uv run python judge.py
```
