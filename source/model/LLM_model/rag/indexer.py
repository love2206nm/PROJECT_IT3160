"""
Hybrid Indexer for O*NET RAG Engine.
Combines Sparse BM25Okapi (exact keyword/tool match) and Dense FAISS (semantic embedding)
with support for persistence in models/faiss_index/.
"""

from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import joblib

try:
    import faiss
except ImportError:
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

SentenceTransformer = None

from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizer tailored for tech resumes and job descriptions.
    Preserves specialized tokens like C++, C#, .NET, Python-3, and removes punctuation.
    """
    text = text.lower()
    # Normalize special programming language tokens
    text = text.replace("c++", "cpp").replace("c#", "csharp").replace(".net", "dotnet")
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\./]+\b", text)
    return tokens


class HybridIndexer:
    """
    Manages both Sparse (BM25) and Dense (FAISS) indices over the 2,601 micro-chunks.
    Supports in-memory querying and on-disk serialization.
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        index_dir: Optional[Path] = None,
    ) -> None:
        self.embedding_model_name = (
            embedding_model_name or settings.RAG_EMBEDDING_MODEL
        )
        self.index_dir = index_dir or settings.FAISS_INDEX_DIR

        self.embedder: Optional[Any] = None
        self.faiss_index: Optional[Any] = None
        self.bm25_index: Optional[Any] = None

        self.chunk_ids: List[str] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.is_indexed: bool = False

    def _get_embedder(self) -> Any:
        """Lazy load SentenceTransformer embedder with single-thread CPU memory optimization."""
        if self.embedder is None:
            try:
                import torch
                torch.set_num_threads(1)
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model '%s' on demand...", self.embedding_model_name)
                self.embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                logger.warning("Could not initialize SentenceTransformer (%s). Dense vector search will fallback.", e)
                self.embedder = None
        return self.embedder

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Builds both BM25 and FAISS dense indices over the provided chunks.
        """
        if not chunks:
            raise ValueError("Cannot build index on empty chunk list.")

        logger.info("Building Hybrid Index for %d micro-chunks...", len(chunks))

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        corpus_texts = [c["content"] for c in chunks]

        # 1. Build Sparse BM25 Index
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is required. Install with 'pip install rank-bm25'.")

        tokenized_corpus = [simple_tokenize(text) for text in corpus_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("Sparse BM25Okapi index built successfully.")

        # 2. Build Dense FAISS Index
        if faiss is None:
            raise ImportError("faiss-cpu is required. Install with 'pip install faiss-cpu'.")

        embedder = self._get_embedder()
        logger.info("Generating dense embeddings for %d texts...", len(corpus_texts))
        embeddings = embedder.encode(
            corpus_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Crucial for Cosine similarity via Inner Product
        ).astype(np.float32)

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)
        logger.info(
            "Dense FAISS IndexFlatIP (dim=%d, count=%d) built successfully.",
            dimension,
            self.faiss_index.ntotal,
        )

        self.is_indexed = True

    def save_index(self, target_dir: Optional[Path] = None) -> None:
        """Serializes FAISS index and BM25 metadata to disk."""
        if not self.is_indexed or self.faiss_index is None:
            raise RuntimeError("Cannot save uninitialized or empty index.")

        out_dir = target_dir or self.index_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss_file = out_dir / "dense_faiss.index"
        meta_file = out_dir / "hybrid_meta.joblib"

        logger.info("Saving FAISS index to '%s'...", faiss_file)
        faiss.write_index(self.faiss_index, str(faiss_file))

        metadata_payload = {
            "bm25_index": self.bm25_index,
            "chunk_ids": self.chunk_ids,
            "embedding_model_name": self.embedding_model_name,
        }
        logger.info("Saving BM25 and chunk metadata to '%s'...", meta_file)
        joblib.dump(metadata_payload, meta_file)

        logger.info("Hybrid Index serialized successfully to '%s'.", out_dir)

    def load_index(self, source_dir: Optional[Path] = None) -> None:
        """Loads FAISS index and BM25 metadata from disk."""
        src_dir = source_dir or self.index_dir
        faiss_file = src_dir / "dense_faiss.index"
        meta_file = src_dir / "hybrid_meta.joblib"

        if not (faiss_file.exists() and meta_file.exists()):
            raise FileNotFoundError(
                f"Index files not found in '{src_dir}'. "
                "Run CLI 'python -m source.services.cli.build_rag_index_cli' first."
            )

        logger.info("Loading FAISS dense index from '%s'...", faiss_file)
        self.faiss_index = faiss.read_index(str(faiss_file))

        logger.info("Loading BM25 index and metadata from '%s'...", meta_file)
        payload = joblib.load(meta_file)
        self.bm25_index = payload["bm25_index"]
        self.chunk_ids = payload["chunk_ids"]
        self.chunk_id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        self.embedding_model_name = payload.get("embedding_model_name", self.embedding_model_name)

        self.is_indexed = True
        logger.info(
            "Hybrid Index loaded successfully (%d chunks, FAISS total=%d).",
            len(self.chunk_ids),
            self.faiss_index.ntotal,
        )

    def search_sparse(
        self, query: str, allowed_chunk_ids: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """
        Queries BM25 index and returns {chunk_id: score} dict for allowed chunks.
        """
        if self.bm25_index is None:
            raise RuntimeError("BM25 index not initialized.")

        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return {}

        scores = self.bm25_index.get_scores(query_tokens)
        results: Dict[str, float] = {}

        for idx, score in enumerate(scores):
            cid = self.chunk_ids[idx]
            if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
                continue
            if score > 0.0:
                results[cid] = float(score)

        return results

    def search_dense(
        self, query: str, allowed_chunk_ids: Optional[Set[str]] = None, top_k: int = 50
    ) -> Dict[str, float]:
        """
        Queries FAISS dense index and returns {chunk_id: cosine_sim} dict for allowed chunks.
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not initialized.")

        embedder = self._get_embedder()
        if embedder is None:
            return {}

        query_vec = embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        # Retrieve top candidates from FAISS
        # When allowed_chunk_ids is restricted, search with larger k then filter
        search_k = min(len(self.chunk_ids), max(top_k * 3, 100))
        sims, indices = self.faiss_index.search(query_vec, search_k)

        results: Dict[str, float] = {}
        for sim, idx in zip(sims[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            cid = self.chunk_ids[idx]
            if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
                continue
            results[cid] = float(sim)

        return results
