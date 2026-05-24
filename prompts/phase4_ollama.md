# Phase 4 Ollama Prompt

PHASE 4 (Ollama): Generate synthesized goldens locally and evaluate with the local Ollama judge.

1. Confirm Phase 2 and Phase 3 completed successfully.
2. Confirm `src/synthesize.py` uses `build_judge_model()` and an explicit `OllamaEmbeddingModel` for local context construction.
3. Confirm `src/eval_synthesized.py` uses `build_lightrag()`, `build_judge_model()`, and the shared provider-switchable config.
4. Run `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/synthesize.py 2>&1 | tee results/synthesize.log`.
5. Verify `tests/synthesized_qa.json` exists with at least 8 entries.
6. Run `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/eval_synthesized.py 2>&1 | tee results/eval_synthesized.log`.
7. Print summary with `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python -c "import json; from collections import defaultdict; data=json.load(open('results/synthesized_scores.json')); agg=defaultdict(list); [agg[m].append(s) for d in data for m,s in d['scores'].items() if s is not None]; print('\\n=== SYNTHESIZED EVAL SUMMARY ==='); [print(f'{m:30s} avg={sum(v)/len(v):.3f}  pass(>=0.7)={sum(1 for x in v if x>=0.7)}/{len(v)}') for m,v in agg.items()]; print(f'\\nTotal cases: {len(data)}')"`.
8. Print exactly: `PHASE 4 OLLAMA COMPLETE. Both eval sets done.`

If any phase trips a new error, stop, paste relevant log lines, and update `learner.md`.
