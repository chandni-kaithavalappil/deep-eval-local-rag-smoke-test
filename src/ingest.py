"""Ingest Markdown files into a LightRAG index under ./rag_storage."""
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightrag.base import DocStatus
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.operate import chunking_by_token_size
from lightrag.utils import compute_mdhash_id
from lightrag_config import build_lightrag, describe_config

DOCS_DIR = Path("docs")
STORAGE_DIR = Path("rag_storage")
DOC_GLOB = os.getenv("RAG_DOC_GLOB", "docs/*.md")
INGEST_MODE = os.getenv("RAG_INGEST_MODE", "graph").lower()


async def naive_insert(rag: Any, path: Path, text: str) -> None:
    """Insert chunks and embeddings without LightRAG's entity/relationship extraction."""
    doc_id = compute_mdhash_id(text, prefix="doc-")
    raw_chunks = chunking_by_token_size(
        rag.tokenizer,
        text,
        chunk_overlap_token_size=rag.chunk_overlap_token_size,
        chunk_token_size=rag.chunk_token_size,
    )
    chunks = {}
    chunk_ids = []
    for chunk in raw_chunks:
        chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")
        chunk_ids.append(chunk_id)
        chunks[chunk_id] = {
            "content": chunk["content"],
            "full_doc_id": doc_id,
            "tokens": chunk["tokens"],
            "chunk_order_index": chunk["chunk_order_index"],
            "file_path": str(path),
        }

    now = datetime.now(timezone.utc).isoformat()
    await asyncio.gather(
        rag.full_docs.upsert({doc_id: {"content": text, "file_path": str(path)}}),
        rag.text_chunks.upsert(chunks),
        rag.chunks_vdb.upsert(chunks),
        rag.doc_status.upsert(
            {
                doc_id: {
                    "content_summary": text[:100],
                    "content_length": len(text),
                    "file_path": str(path),
                    "status": DocStatus.PROCESSED,
                    "created_at": now,
                    "updated_at": now,
                    "chunks_count": len(chunks),
                    "chunks_list": chunk_ids,
                    "metadata": {"ingest_mode": "naive"},
                }
            }
        ),
    )
    await rag._insert_done()


async def main() -> None:
    STORAGE_DIR.mkdir(exist_ok=True)
    print(f"Using {describe_config(STORAGE_DIR)}")
    rag = build_lightrag(STORAGE_DIR)
    await rag.initialize_storages()
    await initialize_pipeline_status()

    md_files = sorted(Path().glob(DOC_GLOB))
    if not md_files:
        raise SystemExit(f"No .md files matched RAG_DOC_GLOB={DOC_GLOB!r}.")

    print(f"Ingesting {len(md_files)} markdown file(s) from {DOC_GLOB!r}...")
    print(f"Using ingest_mode={INGEST_MODE}")
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        print(f"  - {path} ({len(text):,} chars)")
        if INGEST_MODE == "naive":
            await naive_insert(rag, path, text)
        elif INGEST_MODE == "graph":
            await rag.ainsert(text, file_paths=str(path))
        else:
            raise SystemExit("RAG_INGEST_MODE must be 'graph' or 'naive'.")

    print(f"Done. Index written to {STORAGE_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
