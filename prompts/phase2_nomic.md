# Phase 2 Nomic Embeddings Prompt

PHASE 2 FIX (nomic Plan B): Switch local LightRAG embeddings to `nomic-embed-text` after both Qwen embedding models timed out during ingest. Keep `RAG_LLM_PROVIDER=ollama`, `RAG_LLM_MODEL=granite4.1:8b`, `RAG_EMBED_PROVIDER=ollama`, and set `RAG_EMBED_MODEL=nomic-embed-text`. Do not trust the dimension until verified live.

1. Confirm Ollama is reachable and `nomic-embed-text` is installed.
2. Run `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/check_config.py --embed 2>&1 | tee results/check_embed_nomic.log`.
3. If `embedding_dim actual=N configured=M` differs, update `.env` so `RAG_EMBED_DIM=N`, then rerun the embed smoke test until it exits 0.
4. Verify `rag_storage/` is empty. If not, archive it with a timestamp and recreate an empty directory.
5. Run `UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python src/ingest.py 2>&1 | tee results/ingest_nomic.log`.
6. Scan for failures with `grep -nE "Traceback|ERROR|TimeoutError|Failed to extract|Worker timeout" results/ingest_nomic.log`. If any line matches, stop and paste the relevant lines.
7. Verify `rag_storage/` contains populated LightRAG files including `kv_store_*.json`, `vdb_chunks.json`, and `graph_chunk_entity_relation.graphml`.
8. Print exactly: `PHASE 2 OLLAMA COMPLETE. Index built with granite4.1:8b + nomic-embed-text (<actual-dim>-dim).`

After every meaningful result or failure, update `learner.md`.
