"""
CLI Training & Build Pipeline for Hierarchical Aspect-Based RAG Index.
Embeds 2,601 micro-chunks via SentenceTransformer, builds FAISS IndexFlatIP & BM25Okapi,
and serializes artifacts to models/faiss_index/.
Zero print statements - 100% structured logging.
"""

from pathlib import Path
import time
import argparse
from typing import Optional

from source.model.LLM_model.rag.corpus_manager import CorpusManager
from source.model.LLM_model.rag.indexer import HybridIndexer
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


def build_and_serialize_rag_index(
    corpus_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    embedding_model_name: Optional[str] = None,
) -> None:
    """
    Orchestrates RAG index generation:
    1. Loads 2,601 micro-chunks from data/clean_data/final/rag_corpus.json
    2. Builds BM25Okapi sparse index
    3. Generates L2-normalized embeddings via all-MiniLM-L6-v2 and builds FAISS index
    4. Serializes index artifacts into models/faiss_index/
    """
    start_total = time.perf_counter()

    c_path = corpus_path or (settings.DATA_CLEAN_DIR / "final" / "rag_corpus.json")
    out_dir = output_dir or settings.FAISS_INDEX_DIR
    model_name = embedding_model_name or settings.RAG_EMBEDDING_MODEL

    logger.info("=" * 80)
    logger.info("TASK 5: BUILDING HIERARCHICAL ASPECT-BASED RAG INDEX")
    logger.info("Corpus Path    : %s", c_path)
    logger.info("Output Dir     : %s", out_dir)
    logger.info("Embedding Model: %s", model_name)
    logger.info("=" * 80)

    # 1. Load Corpus
    corpus_mgr = CorpusManager(corpus_path=c_path)
    chunks = corpus_mgr.all_chunks
    logger.info("Corpus loaded: %d chunks across %d major groups.", len(chunks), len(corpus_mgr._by_major_group))

    # 2. Build Hybrid Index
    indexer = HybridIndexer(embedding_model_name=model_name, index_dir=out_dir)
    indexer.build_index(chunks)

    # 3. Serialize to disk
    indexer.save_index(target_dir=out_dir)

    # 4. Measure artifact sizes
    faiss_file = out_dir / "dense_faiss.index"
    meta_file = out_dir / "hybrid_meta.joblib"

    faiss_size_kb = faiss_file.stat().st_size / 1024 if faiss_file.exists() else 0.0
    meta_size_kb = meta_file.stat().st_size / 1024 if meta_file.exists() else 0.0
    elapsed_sec = time.perf_counter() - start_total

    logger.info("-" * 80)
    logger.info("RAG INDEX BENCHMARK & SERIALIZATION SUMMARY")
    logger.info("  * Total Chunks Indexed: %d", len(chunks))
    logger.info("  * FAISS Index File    : %s (%.1f KB)", faiss_file, faiss_size_kb)
    logger.info("  * BM25 / Meta File    : %s (%.1f KB)", meta_file, meta_size_kb)
    logger.info("  * Total Elapsed Time  : %.2f seconds", elapsed_sec)
    logger.info("=" * 80)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Build and serialize Hybrid FAISS/BM25 Index for O*NET Knowledge Corpus."
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=str(settings.DATA_CLEAN_DIR / "final" / "rag_corpus.json"),
        help="Path to rag_corpus.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.FAISS_INDEX_DIR),
        help=f"Directory to save FAISS and BM25 index artifacts (default: {settings.FAISS_INDEX_DIR})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=settings.RAG_EMBEDDING_MODEL,
        help="SentenceTransformer model name for dense vector embeddings",
    )

    args = parser.parse_args()

    build_and_serialize_rag_index(
        corpus_path=Path(args.corpus_path),
        output_dir=Path(args.output_dir),
        embedding_model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
