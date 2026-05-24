"""Print and optionally smoke-test the configured LightRAG providers."""
import argparse
import asyncio

from lightrag_config import (
    build_lightrag,
    describe_config,
    get_config,
)
from deepeval_config import (
    build_judge_model,
    describe_judge_model,
    get_judge_provider,
)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", action="store_true", help="run one embedding call")
    parser.add_argument("--llm", action="store_true", help="run one LLM call")
    parser.add_argument("--judge", action="store_true", help="run one DeepEval judge call")
    args = parser.parse_args()

    config = get_config()
    print(describe_config())

    if args.embed:
        rag = build_lightrag()
        vectors = await rag.embedding_func(["configuration smoke test"])
        actual_dim = len(vectors[0])
        print(f"embedding_dim actual={actual_dim} configured={config.embed_dim}")
        if actual_dim != config.embed_dim:
            raise SystemExit(
                "Embedding dimension mismatch. Update RAG_EMBED_DIM or clear rag_storage/ "
                "before ingesting with the corrected value."
            )

    if args.llm:
        if config.llm_provider != "ollama":
            print("Skipping LLM smoke test for API providers to avoid accidental spend.")
            return
        import ollama

        response = ollama.chat(
            model=config.llm_model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            options={"num_predict": 8},
        )
        print(response.message.content.strip())

    if args.judge:
        print(describe_judge_model())
        if get_judge_provider() != "ollama":
            print("Skipping judge smoke test for API providers to avoid accidental spend.")
            return
        judge = build_judge_model()
        response, _ = judge.generate("Reply with exactly: ok")
        print(response.strip())


if __name__ == "__main__":
    asyncio.run(main())
