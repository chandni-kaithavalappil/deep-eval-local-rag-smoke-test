# DeepEval Local RAG Smoke Test

This repo is a small, reproducible experiment for evaluating a local RAG pipeline with
[LightRAG](https://github.com/HKUDS/LightRAG), [DeepEval](https://github.com/confident-ai/deepeval),
and [Ollama](https://ollama.com/).

The original goal was a full local LightRAG graph benchmark over a small markdown corpus. In practice,
local model runtime limits became the story:

- OpenAI-backed ingest was blocked by API quota.
- `qwen3-embedding:4b` and `qwen3-embedding:0.6b` timed out during LightRAG ingest.
- `nomic-embed-text` fixed embeddings, but full graph extraction with `granite4.1:8b` timed out.
- The experiment pivoted to a one-document, vector-only smoke test so DeepEval metrics could be demonstrated reliably.

## Dashboard

GitHub Pages:

```text
https://chandni-kaithavalappil.github.io/deep-eval-local-rag-smoke-test/
```

If the URL returns 404 after pushing, enable it in GitHub:

```text
Repository Settings -> Pages -> Build and deployment
Source: Deploy from a branch
Branch: gh-pages
Folder: / (root)
Save
```

Open the standalone dashboard:

```text
results/deepeval_dashboard.html
```

It summarizes the experiment path, local model bottlenecks, the smoke-test pivot, and the final metric scores.

Current demo scores:

| Metric suite | Avg score | Pass rate | Cases |
| --- | ---: | ---: | ---: |
| Answer Relevancy | 0.611 | 0/3 | 3 |
| Contextual Relevancy | 0.278 | 0/3 | 3 |
| Faithfulness | 0.667 | 0/3 | 3 |
| Refusal Pattern | 1.000 | 1/1 | 1 |

These are smoke-test scores, not a completed full-corpus benchmark.

## What's Included

```text
docs/                         Markdown corpus used for the experiment
prompts/                      Phase prompts and run notes
src/                          LightRAG, DeepEval, and provider configuration scripts
tests/                        Handcrafted and smoke-test QA sets
results/deepeval_dashboard.html
results/demo_*_scores.json    Final demo score artifacts
results/demo_scores_summary.* Consolidated score summary
learner.md                    Experiment log: what worked, what failed, what to avoid
```

Generated local indexes, failed ingest stores, virtual environments, caches, logs, and `.env` are intentionally not committed.

## Requirements

- Python 3.10, 3.11, or 3.12
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com/) running locally

Pull the models used for the smoke test:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
UV_CACHE_DIR="$PWD/.uv-cache" uv pip install -e .
cp .env.example .env
```

The `.env.example` defaults are set up for the working smoke-test path:

```bash
RAG_LLM_PROVIDER=ollama
RAG_LLM_MODEL=llama3.2:1b
RAG_EMBED_PROVIDER=ollama
RAG_EMBED_MODEL=nomic-embed-text
RAG_EMBED_DIM=768
DEEPEVAL_PROVIDER=ollama
DEEPEVAL_MODEL=llama3.2:1b
```

Confirm local configuration:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/check_config.py --embed --llm --judge
```

## Rebuild the Smoke Index

The committed repo does not include `rag_storage/`. Rebuild it locally:

```bash
RAG_INGEST_MODE=naive \
RAG_DOC_GLOB="docs/doc_02.md" \
UV_CACHE_DIR="$PWD/.uv-cache" \
uv run --no-sync python src/ingest.py
```

This builds a vector-only index for `docs/doc_02.md` and skips LightRAG entity/relation graph extraction.

## Run the Demo Metrics

Use one metric at a time. This is slower but much more reliable with a small local Ollama judge.

Answer relevancy:

```bash
RAG_LLM_MODEL="llama3.2:1b" \
DEEPEVAL_MODEL="llama3.2:1b" \
RAG_QUERY_MODE=naive \
QA_PATH="tests/smoke_qa_doc_02.json" \
RESULTS_PATH="results/demo_answer_relevancy_scores.json" \
DEEPEVAL_CASE_LIMIT=3 \
DEEPEVAL_METRICS=answer_relevancy \
DEEPEVAL_INCLUDE_REASON=false \
DEEPEVAL_MAX_CONCURRENT=1 \
DEEPEVAL_RETRY_MAX_ATTEMPTS=1 \
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=240 \
DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=900 \
UV_CACHE_DIR="$PWD/.uv-cache" \
uv run --no-sync python src/eval_handcrafted.py
```

Contextual relevancy:

```bash
RAG_LLM_MODEL="llama3.2:1b" \
DEEPEVAL_MODEL="llama3.2:1b" \
RAG_QUERY_MODE=naive \
QA_PATH="tests/smoke_qa_doc_02.json" \
RESULTS_PATH="results/demo_contextual_relevancy_scores.json" \
DEEPEVAL_CASE_LIMIT=3 \
DEEPEVAL_METRICS=contextual_relevancy \
DEEPEVAL_INCLUDE_REASON=false \
DEEPEVAL_CONTEXT_CHAR_LIMIT=2500 \
DEEPEVAL_MAX_TOKENS=4096 \
DEEPEVAL_MAX_CONCURRENT=1 \
DEEPEVAL_RETRY_MAX_ATTEMPTS=1 \
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300 \
DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=1200 \
UV_CACHE_DIR="$PWD/.uv-cache" \
uv run --no-sync python src/eval_handcrafted.py
```

Faithfulness:

```bash
RAG_LLM_MODEL="llama3.2:1b" \
DEEPEVAL_MODEL="llama3.2:1b" \
RAG_QUERY_MODE=naive \
QA_PATH="tests/smoke_qa_doc_02.json" \
RESULTS_PATH="results/demo_faithfulness_scores.json" \
DEEPEVAL_CASE_LIMIT=3 \
DEEPEVAL_METRICS=faithfulness \
DEEPEVAL_INCLUDE_REASON=false \
DEEPEVAL_CONTEXT_CHAR_LIMIT=2500 \
DEEPEVAL_MAX_TOKENS=4096 \
DEEPEVAL_MAX_CONCURRENT=1 \
DEEPEVAL_RETRY_MAX_ATTEMPTS=1 \
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=300 \
DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=1200 \
UV_CACHE_DIR="$PWD/.uv-cache" \
uv run --no-sync python src/eval_handcrafted.py
```

Adversarial refusal pattern:

```bash
RAG_LLM_MODEL="llama3.2:1b" \
RAG_QUERY_MODE=naive \
QA_PATH="tests/smoke_adversarial_doc_02.json" \
RESULTS_PATH="results/demo_refusal_pattern_scores.json" \
DEEPEVAL_METRICS=refusal_pattern \
DEEPEVAL_CONTEXT_CHAR_LIMIT=2500 \
UV_CACHE_DIR="$PWD/.uv-cache" \
uv run --no-sync python src/eval_handcrafted.py
```

## Regenerate the Score Summary

After rerunning metrics, regenerate the consolidated summary:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/build_demo_summary.py
```

## Notes and Caveats

- `results/deepeval_dashboard.html` is static. If you rerun metrics, update the dashboard values manually or regenerate it.
- The local judge used here is intentionally small. It makes the demo accessible, but it can emit malformed structured JSON on larger prompts.
- `DEEPEVAL_CONTEXT_CHAR_LIMIT=2500` was needed to make context-heavy metrics complete reliably.
- The full LightRAG graph/corpus benchmark is not complete in this repo; see `learner.md` for the detailed trail.
