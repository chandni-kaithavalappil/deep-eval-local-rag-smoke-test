# learner.md

## Project Context
- Sunday experiment for evaluating a LightRAG pipeline with DeepEval.
- Key folders: `docs/` holds fetched markdown corpus, `src/` holds ingest/eval scripts, `tests/` holds handcrafted QA, `rag_storage/` holds generated LightRAG index files, `results/` holds eval outputs.
- Original scaffold used OpenAI for LightRAG generation, embeddings and DeepEval judging. The repo is being adapted to a free/local Ollama path for LightRAG.
- This directory is not currently a git repository.

## Working Rules
- DeepEval judge model selection is centralized in `src/deepeval_config.py`; keep it separate from LightRAG answer-generation config.
- LightRAG provider/model selection is centralized in `src/lightrag_config.py`; do not hardcode providers in ingest/eval scripts.
- Verify a new embedding model with `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/check_config.py --embed` before ingesting. Update `RAG_EMBED_DIM` if the actual dimension differs.
- Current preferred free/local model direction as of May 2026: use `granite4.1:8b` for LightRAG extraction/generation. `qwen3-embedding:4b` returns 2560 dimensions but times out during ingest on this hardware; `qwen3-embedding:0.6b` returns 1024 dimensions but also timed out during ingest under LightRAG defaults. `nomic-embed-text` returns 768 dimensions and clears the embedding smoke test, but the full ingest then times out in Granite entity/relation extraction.
- Keep `learner.md` updated after every meaningful setup/config/run attempt, including failed attempts, verified dimensions, archived storage directories, and the next safe branch.
- Do not record API keys or credentials in logs, output summaries, or `learner.md`.
- Keep `UV_CACHE_DIR="$PWD/.uv-cache"` on `uv` commands for this project.
- Prefer `uv run --no-sync ...` after dependencies are installed; plain `uv run` may try to resolve package metadata from PyPI.
- Treat `rag_storage/` as model-specific. Do not reuse an index created with one embedding dimension/provider after switching embeddings.
- LightRAG ingest can print `Done` even after logging document extraction errors; verify storage files and scan output before declaring Phase 2 complete.
- For local LightRAG, Ollama must be running and accessible at `http://localhost:11434`.
- Phase 2 is still incomplete as of the `nomic-embed-text` attempt. Do not proceed to Phase 3 unless `rag_storage/` contains a populated, clean index and the ingest log has no `Traceback`, `ERROR`, `TimeoutError`, `Failed to extract`, or `Worker timeout` lines.
- For a fast DeepEval metrics demo, use vector-only smoke mode instead of the full corpus graph build: set `RAG_INGEST_MODE=naive`, `RAG_DOC_GLOB=docs/doc_02.md`, `RAG_QUERY_MODE=naive`, `QA_PATH=tests/smoke_qa_doc_02.json`, and `RESULTS_PATH=results/smoke_doc_02_scores.json`.
- The current `rag_storage/` is a valid vector-only smoke index for `docs/doc_02.md`, not a full graph/corpus Phase 2 index. It contains 1 full doc, 2 text chunks, populated `vdb_chunks.json`, and intentionally empty entity/relation vector DBs/graph storage.
- For local DeepEval demos on this hardware, start with `DEEPEVAL_CASE_LIMIT=1` and a small `DEEPEVAL_METRICS` set. Broad local judge runs such as 3 cases x 5 metrics can stall for many minutes or hit retry timeouts.
- `DEEPEVAL_METRICS` now supports `answer_relevancy`, `faithfulness`, `contextual_relevancy`, `contextual_recall`, `refusal`, `refusal_pattern`, and `exact_match` for handcrafted eval. Synthesized eval supports the same set minus `refusal` and `refusal_pattern`.
- Use `DEEPEVAL_CONTEXT_CHAR_LIMIT=2500` for local demo metrics that inspect retrieval context; this made `contextual_relevancy` complete cleanly after an uncapped run produced truncated/invalid JSON from the small local judge.
- Local Ollama access may fail under the command sandbox with `Failed to connect to Ollama`; rerun important local Ollama commands with approval/local access before assuming Ollama itself is down.

## Build, Test and Run Commands
- Worked with approval/network: `ollama pull granite4.1:8b`
- Worked with approval/network: `ollama pull qwen3-embedding:4b`
- Worked with approval/network: `ollama pull qwen3-embedding:0.6b`
- Worked with approval/local access: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/check_config.py --embed` confirmed `qwen3-embedding:4b` returns 2560 dimensions.
- Worked with approval/local access: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/check_config.py --embed` confirmed `qwen3-embedding:0.6b` returns 1024 dimensions after setting `RAG_EMBED_DIM=1024`.
- Worked with approval/local access: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/check_config.py --embed` confirmed `nomic-embed-text` returns 768 dimensions after setting `RAG_EMBED_MODEL=nomic-embed-text` and `RAG_EMBED_DIM=768`.
- Worked with approval/local access: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/check_config.py --llm` confirmed `granite4.1:8b` responds.
- Worked with approval/local access: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/check_config.py --embed --llm --judge` confirmed local embeddings, LightRAG LLM, and DeepEval judge config.
- Worked: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python -m py_compile src/lightrag_config.py src/deepeval_config.py src/local_lightrag.py src/ingest.py src/eval_handcrafted.py src/check_config.py`
- Worked: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python -m py_compile src/synthesize.py src/eval_synthesized.py`
- Worked: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python -m py_compile src/ingest.py src/eval_handcrafted.py src/eval_synthesized.py`
- Worked: `RAG_INGEST_MODE=naive RAG_DOC_GLOB="docs/doc_02.md" UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/ingest.py` built the one-doc vector-only smoke index with `nomic-embed-text` 768-dim embeddings. Output log: `results/ingest_smoke_doc_02_naive.log`.
- Worked with approval/local access: `RAG_LLM_MODEL="llama3.2:1b" RAG_QUERY_MODE=naive QA_PATH="tests/smoke_qa_doc_02.json" RESULTS_PATH="results/smoke_doc_02_exact_scores.json" DEEPEVAL_CASE_LIMIT=3 DEEPEVAL_METRICS=exact_match UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/eval_handcrafted.py` produced a fast DeepEval display run. Exact Match scored 0.0 on all three smoke cases, which is expected for free-form RAG answers versus concise expected answers.
- Worked with approval/local access: `RAG_LLM_MODEL="llama3.2:1b" DEEPEVAL_MODEL="llama3.2:1b" RAG_QUERY_MODE=naive QA_PATH="tests/smoke_qa_doc_02.json" RESULTS_PATH="results/smoke_doc_02_answer_relevancy_scores.json" DEEPEVAL_CASE_LIMIT=1 DEEPEVAL_METRICS=answer_relevancy DEEPEVAL_INCLUDE_REASON=false DEEPEVAL_RETRY_MAX_ATTEMPTS=1 DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=180 UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/eval_handcrafted.py` completed in ~61s and scored Answer Relevancy 0.667 on the first ASL-3 smoke case, below the 0.7 threshold.
- Worked with approval/local access on 2026-05-25: `DEEPEVAL_METRICS=answer_relevancy` over all 3 smoke cases completed in ~155s. Scores: 0.667, 0.667, 0.500; average 0.611; output `results/demo_answer_relevancy_scores.json`.
- Worked with approval/local access on 2026-05-25 after setting `DEEPEVAL_CONTEXT_CHAR_LIMIT=2500` and `DEEPEVAL_MAX_TOKENS=4096`: `DEEPEVAL_METRICS=contextual_relevancy` over all 3 smoke cases completed in ~281s. Scores: 0.333, 0.500, 0.000; average 0.278; output `results/demo_contextual_relevancy_scores.json`.
- Worked with approval/local access on 2026-05-25: `DEEPEVAL_METRICS=faithfulness` over all 3 smoke cases completed in ~750s. Scores: 0.667, 0.667, 0.667; average 0.667; output `results/demo_faithfulness_scores.json`.
- Worked with approval/local access on 2026-05-25: `DEEPEVAL_METRICS=refusal_pattern` on `tests/smoke_adversarial_doc_02.json` completed quickly and scored 1.000; output `results/demo_refusal_pattern_scores.json`.
- Worked on 2026-05-25: generated consolidated demo artifacts `results/demo_scores_summary.json` and `results/demo_scores_summary.md`.
- Worked on 2026-05-25: created a self-contained visual dashboard at `results/deepeval_dashboard.html` that tells the full experiment story: original full LightRAG goal, local timeout/quota blockers, one-doc vector-only pivot, model path, metric cards, case heatmap, and demo takeaways. Verified desktop and mobile render through a temporary local static server; no horizontal overflow after adding wrapping rules for inline code and labels.
- Worked on 2026-05-25: packaged the public GitHub repo as `chandni-kaithavalappil/deep-eval-local-rag-smoke-test`. Curated commit includes source, docs, prompts, tests, smoke dashboard, summary score artifacts, `.env.example`, `uv.lock`, and this learning log. Excluded `.env`, local caches, virtualenv, `.deepeval`, generated `rag_storage*`, old verbose logs, and unrelated LinkedIn draft artifacts.
- Worked on 2026-05-25: added GitHub Pages support for the dashboard. Root `index.html` redirects to `results/deepeval_dashboard.html`; because the GitHub Actions Pages workflow failed before Pages was enabled and no `gh` CLI/API auth was available locally, the fallback is to publish the static content through a `gh-pages` branch. Expected URL: `https://chandni-kaithavalappil.github.io/deep-eval-local-rag-smoke-test/`.
- Worked: `ollama pull llama3.2:1b` installed a smaller local model for smoke answer generation/judging experiments.
- Abandoned slow graph smoke ingest: one-doc `docs/doc_02.md` ingest with `MAX_ASYNC=1`, `MAX_PARALLEL_INSERT=1`, and `LLM_TIMEOUT=900` still stayed in Granite entity extraction for many minutes; archived partial storage as `rag_storage_smoke_graph_abandoned_20260517_195420/`.
- Worked: `UV_CACHE_DIR="$PWD/.uv-cache" uv venv --python 3.11`
- Worked: `UV_CACHE_DIR="$PWD/.uv-cache" uv pip install -e .`
- Worked with approval/network: `bash docs/fetch_corpus.sh`
- Worked with approval/local access: `ollama list`
- Worked with approval/network: `ollama pull nomic-embed-text`
- Worked with approval/network: `UV_CACHE_DIR="$PWD/.uv-cache" uv pip install ollama`
- Worked: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python -c "from local_lightrag import embedding_func; print(embedding_func.embedding_dim)"`
- Failed without network/sandbox approval: `uv run` dependency checks and corpus fetches that need DNS/network.
- Failed with OpenAI quota: `UV_CACHE_DIR="$PWD/.uv-cache" uv run python src/ingest.py` using OpenAI embeddings.
- Failed with embedding worker timeout: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/ingest.py` using `qwen3-embedding:4b`; archived partial storage as `rag_storage_qwen4b_failed_20260517_184406/`.
- Failed with embedding worker timeout: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/ingest.py` using `qwen3-embedding:0.6b`; archived partial storage as `rag_storage_qwen0p6b_failed_20260517_184742/`.
- Failed with LLM extraction worker timeout: `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/ingest.py` using `granite4.1:8b` + `nomic-embed-text`; embeddings cleared, but entity/relation extraction timed out after 360s. Archived partial storage as `rag_storage_nomic_llm_failed_20260517_191048/`.
- Failed/too slow: 3-case x 5-metric smoke eval using `llama3.2:1b` as judge timed out repeatedly at 600s per attempt. Do not use the full local metric suite for quick demos.
- Failed: `llama3.2:1b` judge with the full metric suite produced invalid/truncated JSON for Contextual Recall in one retry path. Keep `include_reason=false`, one metric, and one case when testing small local judges.
- Failed on 2026-05-25: uncapped `contextual_relevancy` over 3 smoke cases produced invalid/truncated JSON from `llama3.2:1b`; retrying with `DEEPEVAL_CONTEXT_CHAR_LIMIT=2500` and `DEEPEVAL_MAX_TOKENS=4096` succeeded.
- Misleading on 2026-05-25: LLM-judged `refusal` GEval on the adversarial case scored 0.000 even though the answer clearly said it lacked enough information. Prefer `refusal_pattern` for the local demo, and treat local GEval refusal scores as judge-quality caveats.
- Failed under sandbox/local network restrictions: local Ollama eval reported `Failed to connect to Ollama`; the same run worked after rerunning with approval/local access.

## Corrections and Learnings
- Situation: User clarified that the near-term goal is to see how DeepEval shows metrics, not necessarily to complete a full six-document local RAG build.
  Mistake / Gap: The previous path treated full-corpus LightRAG ingest as mandatory, which exposed useful local-model bottlenecks but blocked the DeepEval metrics demo.
  Correction: Added smoke-mode env hooks: `RAG_DOC_GLOB` for selecting docs during ingest, `QA_PATH` and `RESULTS_PATH` for selecting eval inputs/outputs, `RAG_QUERY_MODE` for eval retrieval mode, plus `tests/smoke_qa_doc_02.json` focused on the Responsible Scaling Policy.
  Future Rule: When the goal is metric demonstration, prefer the smallest corpus and QA set that exercises answerable and adversarial cases; reserve full-corpus ingest for a later robustness run.

- Situation: The first practical DeepEval demo needed to show metric output without waiting for the full local judge suite.
  Mistake / Gap: The initial smoke eval still ran all five LLM-judged metrics across three cases, causing timeouts and malformed JSON from the tiny local judge.
  Correction: Added `DEEPEVAL_CASE_LIMIT`, `DEEPEVAL_METRICS`, `DEEPEVAL_INCLUDE_REASON`, and `DEEPEVAL_ASYNC_MODE` switches to `src/eval_handcrafted.py` and `src/eval_synthesized.py`; added `ExactMatchMetric` as a fast deterministic DeepEval metric; then ran `exact_match` over 3 smoke cases and `answer_relevancy` over 1 smoke case.
  Future Rule: For demos, first prove the DeepEval reporting loop with `exact_match`; then add one LLM-judged RAG metric at a time, starting with `answer_relevancy`.

- Situation: User wanted more DeepEval scores that make sense for a demo.
  Mistake / Gap: Exact Match was technically useful but semantically unhelpful for free-form RAG answers, and LLM-judged refusal GEval was unreliable with the tiny local judge.
  Correction: Generated a compact demo score set: Answer Relevancy avg 0.611 over 3 cases, Contextual Relevancy avg 0.278 over 3 cases, Faithfulness avg 0.667 over 3 cases, and Refusal Pattern 1.000 on the adversarial case. Wrote `results/demo_scores_summary.md` and `results/demo_scores_summary.json`.
  Future Rule: For article/demo purposes, lead with the completed demo summary artifact and explain that scores are intentionally from a one-doc vector-only smoke index, not the full LightRAG graph benchmark.

- Situation: User wanted the scores in a visual dashboard telling the whole experiment story.
  Mistake / Gap: The score summary existed as Markdown/JSON, but it did not communicate the narrative arc or make the metric contrast visually obvious.
  Correction: Added `results/deepeval_dashboard.html`, a standalone dashboard with status chips, stack summary, metric cards, experiment timeline, comparison bars, case-level heatmap, model-path table, and takeaways. Verified it at desktop and mobile widths.
  Future Rule: Use the HTML dashboard as the primary visual artifact for sharing the demo, and keep it tied to the smoke-demo caveat that the full graph/corpus benchmark is not complete.

- Situation: User wanted the relevant material pushed to GitHub under `deep-eval-local-rag-smoke-test`.
  Mistake / Gap: The workspace contained many local-only artifacts that should not be public, including `.env`, caches, venv, `.deepeval`, generated vector stores, failed ingest stores, and old logs.
  Correction: Rewrote `.gitignore`, updated README and `.env.example` for the working smoke path, added `src/build_demo_summary.py`, refreshed `uv.lock`, initialized git, staged only the curated public files, and left unrelated LinkedIn files untracked.
  Future Rule: Public commits for this project should stay focused on reproducible source/data/results and exclude runtime artifacts unless explicitly needed for a release.

- Situation: User wanted the HTML dashboard made into GitHub Pages.
  Mistake / Gap: The repo had a standalone dashboard under `results/`, but no Pages entrypoint or deployment config.
  Correction: Added root `index.html` as the Pages entrypoint and README link. Initial GitHub Actions Pages workflow failed at `Configure Pages` because Pages was not enabled, so use branch-based Pages publishing via `gh-pages`.
  Future Rule: If `gh-pages` does not auto-enable the project site, enable Pages manually in repo settings with source `Deploy from a branch`, branch `gh-pages`, folder `/`.

- Situation: Small local Ollama judges can emit malformed structured JSON for context-heavy DeepEval metrics.
  Mistake / Gap: Contextual Relevancy initially used the full retrieval context and failed with truncated JSON.
  Correction: Added `DEEPEVAL_CONTEXT_CHAR_LIMIT` to cap retrieval context passed to DeepEval metrics, then reran Contextual Relevancy with `DEEPEVAL_CONTEXT_CHAR_LIMIT=2500` and `DEEPEVAL_MAX_TOKENS=4096`.
  Future Rule: For local DeepEval runs, cap context first and only increase scope after one metric completes cleanly.

- Situation: `ExactMatchMetric` did not expose the same `.name` attribute as the LLM-judged metrics.
  Mistake / Gap: The JSON summary/export path assumed all DeepEval metric result objects had `.name`.
  Correction: Added a `metric_name()` helper that falls back to `.__name__` or the class name.
  Future Rule: Use defensive metric-name extraction when mixing DeepEval metric families.

- Situation: Even one-doc LightRAG graph ingest was too slow because the built-in insert path always schedules entity/relation extraction.
  Mistake / Gap: Reducing docs alone was not enough if the graph extraction path still uses slow local Granite calls.
  Correction: Added `RAG_INGEST_MODE=naive`, which directly inserts full docs, text chunks, chunk vectors, and processed doc status while skipping entity/relation extraction. Pair it with `RAG_QUERY_MODE=naive` for vector-only retrieval.
  Future Rule: Use naive ingest/query for DeepEval metrics demos; use graph ingest only when evaluating LightRAG's knowledge-graph behavior specifically.

- Situation: Plan B switched embeddings from Qwen to `nomic-embed-text` to unblock Phase 2.
  Mistake / Gap: The embedding bottleneck was fixed, but the full ingest still failed because local `granite4.1:8b` entity/relation extraction exceeded LightRAG's 360s worker timeout for the first chunk batch.
  Correction: Updated `.env` to `RAG_EMBED_MODEL=nomic-embed-text` and `RAG_EMBED_DIM=768`, confirmed the live dimension, ran ingest, stopped on `LLM func: Worker execution timeout after 360s`, archived the partial store, and recreated empty `rag_storage/`.
  Future Rule: The next Phase 2 branch should target LLM extraction, not embeddings. Concrete options are to tune LightRAG LLM/chunking env vars (`LLM_TIMEOUT`, `MAX_ASYNC`, `CHUNK_SIZE`, `CHUNK_OVERLAP_SIZE`) or switch `RAG_LLM_MODEL` to a faster local model before rebuilding from empty `rag_storage/`.

- Situation: Phase 4 prompts were provided for Ollama synthesis and synthesized eval.
  Mistake / Gap: `src/synthesize.py` and `src/eval_synthesized.py` still hardcoded OpenAI paths.
  Correction: Updated synthesis to use `build_judge_model()` plus explicit Ollama embedding context construction, and updated synthesized eval to use `build_lightrag()` and `build_judge_model()`.
  Future Rule: Keep Phase 4 local/Ollama unless the user explicitly permits OpenAI/Anthropic spend.

- Situation: Phase 2 fallback from `qwen3-embedding:4b` to `qwen3-embedding:0.6b` was tested after 4B exceeded LightRAG's 60s embedding worker timeout.
  Mistake / Gap: The fallback model was assumed likely to clear the timeout, but the actual hardware/run still timed out under LightRAG defaults.
  Correction: Pulled `qwen3-embedding:0.6b`, verified its live embedding dimension is 1024, updated `.env`, ran ingest, stopped after repeated `Embedding func: Worker execution timeout after 60s`, and archived the partial `rag_storage/`.
  Future Rule: Do not retry Qwen embeddings for Phase 2 on this hardware unless the user explicitly wants a performance experiment. The next branch should target the LLM extraction timeout discovered after switching to `nomic-embed-text`.

- Situation: Phase 3 was requested before local-Ollama Phase 2 ingestion had produced index files.
  Mistake / Gap: `rag_storage/` was empty even though model config smoke tests had passed.
  Correction: Stop before handcrafted eval when `rag_storage/` lacks expected LightRAG files.
  Future Rule: Phase 3 requires non-empty `rag_storage/` containing files such as `kv_store_*.json`, `vdb_chunks.json`, and `graph_chunk_entity_relation.graphml`.

- Situation: User wanted future experiments to switch easily between local, OpenAI, and Claude models.
  Mistake / Gap: The earlier local-only wiring made provider changes require source edits and risked reusing incompatible `rag_storage/`.
  Correction: Added `src/lightrag_config.py` with env-driven `RAG_LLM_PROVIDER`, `RAG_LLM_MODEL`, `RAG_EMBED_PROVIDER`, `RAG_EMBED_MODEL`, and `RAG_EMBED_DIM`; kept `src/local_lightrag.py` as a compatibility shim.
  Future Rule: Add new providers in `src/lightrag_config.py`, document the env values, and run `src/check_config.py --embed` before ingestion.

- Situation: DeepEval judge config also needed local/OpenAI/Claude switching.
  Mistake / Gap: `eval_handcrafted.py` only passed a model string, which made the provider implicit and OpenAI-biased.
  Correction: Added `src/deepeval_config.py` with `DEEPEVAL_PROVIDER=ollama|openai|anthropic`; `src/check_config.py --judge` validates local judge wiring without API spend.
  Future Rule: Switch DeepEval judges with `DEEPEVAL_PROVIDER` and `DEEPEVAL_MODEL`, not by editing eval scripts.

- Situation: DeepEval's installed Ollama model metadata did not yet include `granite4.1:8b`.
  Mistake / Gap: Unknown Ollama judge models can lack capability metadata for structured-output checks.
  Correction: Added `DEEPEVAL_OLLAMA_CAPABILITY_MODEL=granite4` so the real model can remain `granite4.1:8b` while DeepEval uses compatible known capabilities.
  Future Rule: For newer Ollama judge tags missing from DeepEval constants, set a compatible `DEEPEVAL_OLLAMA_CAPABILITY_MODEL`.

- Situation: Installed `qwen3-embedding:4b` locally.
  Mistake / Gap: Embedding dimensions were initially based on model docs/assumptions.
  Correction: Verified the live Ollama embedding output returns 2560 dimensions.
  Future Rule: Never trust assumed embedding dimensions; smoke-test the actual configured provider and model.

- Situation: User asked whether the initially selected free model was the best fit.
  Mistake / Gap: `hermes3:8b` was selected opportunistically because it was already installed, not because it was the strongest current fit.
  Correction: Research current Ollama-local options before continuing; prefer a newer, retrieval/RAG-suitable model pair. May 2026 check favors `granite4.1:8b` over `qwen3.5:9b` for this repo because Granite 4.1 is newer and explicitly supports RAG, text extraction, structured JSON output, and QA.
  Future Rule: For model selection, optimize for RAG extraction/query quality, structured instruction following, local availability, context length, and practical runtime size.

- Situation: User asked to create and maintain a learning log.
  Mistake / Gap: No project memory file existed for repeated setup failures and corrections.
  Correction: Added this `learner.md` and will review/update it at the end of tasks.
  Future Rule: Before future code changes, read `learner.md` and apply relevant project rules.

- Situation: OpenAI-backed ingest failed during embeddings.
  Mistake / Gap: A real API key was present, but the project/account had `insufficient_quota`.
  Correction: Move LightRAG ingest toward local Ollama models instead of OpenAI.
  Future Rule: Do not retry OpenAI ingest for this repo unless the user confirms quota/billing is fixed.

- Situation: Failed OpenAI and Ollama attempts wrote partial files into `rag_storage/`.
  Mistake / Gap: Partial storage can look superficially valid and may have the wrong embedding dimension.
  Correction: Archived failed directories as `rag_storage_openai_failed_20260517/` and `rag_storage_ollama_failed_20260517/`, then recreated empty `rag_storage/`.
  Future Rule: Start from an empty `rag_storage/` after changing embedding providers or after extraction errors.

- Situation: Local Ollama migration initially bound `model` with `partial(ollama_model_complete, model=...)`.
  Mistake / Gap: LightRAG also passes the model internally, producing `_ollama_model_if_cache() got multiple values for argument 'model'`.
  Correction: Let `LightRAG` own `llm_model_name`; pass only host/options via `llm_model_kwargs`.
  Future Rule: For LightRAG Ollama, use `llm_model_func=ollama_model_complete`, `llm_model_name=<model>`, and `llm_model_kwargs={"host": ...}`.

## Do Not Repeat
- Do not assume a Claude/Anthropic setting covers embeddings; this LightRAG setup supports Anthropic for generation/extraction only, with embeddings from Ollama or OpenAI.
- Do not assume `qwen3-embedding:0.6b` fixes the local ingest timeout; it smoke-tests successfully but still hit LightRAG's 60s embedding worker timeout on ingest in this workspace.
- Do not assume `nomic-embed-text` completes Phase 2; it fixes embeddings but exposed a Granite LLM extraction timeout during ingest.
- Do not present the current one-doc vector-only `rag_storage/` as a completed full Phase 2 graph/corpus index. It is valid for smoke metric demos only.
- Do not start local demo evals with all five LLM-judged metrics. Start with `DEEPEVAL_METRICS=exact_match` or one LLM metric plus `DEEPEVAL_CASE_LIMIT=1`.
- Do not use uncapped retrieval context for local context-heavy metrics if `llama3.2:1b` is the judge; use `DEEPEVAL_CONTEXT_CHAR_LIMIT=2500`.
- Do not use the LLM-judged `refusal` GEval score as the main demo refusal result for this local setup; use `refusal_pattern` unless testing judge reliability itself.
- Do not interpret sandboxed `Failed to connect to Ollama` as conclusive evidence that Ollama is down; rerun with approved local access.
- Do not use `DEEPEVAL_MODEL` alone as proof the judge provider changed; set `DEEPEVAL_PROVIDER` too.
- Do not edit `src/ingest.py` or `src/eval_handcrafted.py` just to switch models; use `.env` and `src/lightrag_config.py`.
- Do not paste or log `.env` secrets.
- Do not declare Phase 2 complete just because `src/ingest.py` prints `Done`; check for extraction tracebacks and non-empty index files.
- Do not reuse `rag_storage/` across embedding dimensions such as OpenAI 1536 and Ollama `nomic-embed-text` 768.
- Do not leave failed ingest partials in `rag_storage/`; archive them and recreate an empty directory before the next ingest attempt.
- Do not bind the Ollama LLM model twice with both `partial(..., model=...)` and LightRAG `llm_model_name`.
- Do not assume `uv run` is offline-safe after `pyproject.toml` changes; use `uv pip install -e .` first, then `uv run --no-sync`.

## Implementation Patterns
- Centralize provider wiring in `src/lightrag_config.py`; ingest and eval should call `build_lightrag()` and print `describe_config()`.
- Use `.env` to switch between `RAG_LLM_PROVIDER=ollama|openai|anthropic` and `RAG_EMBED_PROVIDER=ollama|openai`.
- Centralize DeepEval judge wiring in `src/deepeval_config.py`; metrics should use `build_judge_model()`.
- Use `src/check_config.py --embed --llm --judge` for local Ollama smoke tests.
- Keep failed storage directories archived with explicit names instead of deleting them unless the user asks.

## Open Questions
- Full Phase 3 still needs a reliable free/local DeepEval judge strategy for the complete metric suite. The practical smoke strategy now works with one-file vector retrieval plus one metric at a time; full-suite local judging likely needs a faster/more reliable local model, smaller prompts, or explicit permission to use a paid/cloud judge.
