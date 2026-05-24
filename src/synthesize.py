"""Use DeepEval's Synthesizer to generate ~15 goldens from docs/."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval.models import OllamaEmbeddingModel
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.synthesizer import Synthesizer

from deepeval_config import build_judge_model, describe_judge_model
from lightrag_config import get_config

load_dotenv()

DOCS_DIR = Path("docs")
OUT_PATH = Path("tests/synthesized_qa.json")
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def main() -> None:
    doc_paths = sorted(str(p) for p in DOCS_DIR.glob("*.md"))
    if not doc_paths:
        raise SystemExit(f"No .md files in {DOCS_DIR}/")

    config = get_config()
    if config.embed_provider != "ollama":
        raise SystemExit("Local synthesis expects RAG_EMBED_PROVIDER=ollama.")

    judge_model = build_judge_model()
    print(f"Using {describe_judge_model()} for synthesis")
    print(f"Using ollama embedding model {config.embed_model} for context construction")

    context_config = ContextConstructionConfig(
        embedder=OllamaEmbeddingModel(
            model=config.embed_model,
            base_url=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        ),
        chunk_size=1024,
        chunk_overlap=64,
    )
    synth = Synthesizer(model=judge_model)
    goldens = synth.generate_goldens_from_docs(
        document_paths=doc_paths,
        max_goldens_per_context=2,
        context_construction_config=context_config,
    )

    out = [
        {
            "question": g.input,
            "expected": g.expected_output or "",
            "bucket": "synthesized",
        }
        for g in goldens
    ]
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Generated {len(out)} goldens -> {OUT_PATH}")


if __name__ == "__main__":
    main()
