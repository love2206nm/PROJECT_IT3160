"""
Hierarchical Aspect-Based Evidence Retriever.
Implements IEvidenceRetriever domain protocol using 3-stage retrieval:
1. Pre-filtering Namespace (major_group, aspect, soc_code)
2. Hybrid Search (BM25Okapi + FAISS Dense)
3. Cross-Encoder Reranking
"""

import time
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

CrossEncoder = None

from source.domain.interfaces.rag import IEvidenceRetriever
from source.model.LLM_model.rag.corpus_manager import CorpusManager
from source.model.LLM_model.rag.indexer import HybridIndexer
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class HierarchicalEvidenceRetriever(IEvidenceRetriever):
    """
    Production-grade Hierarchical Aspect-Based RAG Retriever.
    Provides highly grounded factual O*NET evidence for Copilot Agent.
    """

    def __init__(
        self,
        corpus_manager: Optional[CorpusManager] = None,
        indexer: Optional[HybridIndexer] = None,
        use_reranker: Optional[bool] = None,
        reranker_model_name: Optional[str] = None,
    ) -> None:
        self.corpus_manager = corpus_manager or CorpusManager()
        self.indexer = indexer or HybridIndexer()
        self.use_reranker = (
            use_reranker if use_reranker is not None else settings.RAG_USE_RERANKER
        )
        self.reranker_model_name = (
            reranker_model_name or settings.RAG_RERANKER_MODEL
        )
        self.reranker: Optional[Any] = None

        # Automatically load index if on disk and not yet indexed
        if not self.indexer.is_indexed:
            try:
                self.indexer.load_index()
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning(
                    "Hybrid index not found on disk (%s). Auto-building in-memory from corpus...",
                    e,
                )
                self.indexer.build_index(self.corpus_manager.all_chunks)

    def _get_reranker(self) -> Optional[Any]:
        """Lazy load CrossEncoder reranker model on demand."""
        if self.use_reranker and self.reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Initializing CrossEncoder reranker '%s' on demand...", self.reranker_model_name)
                self.reranker = CrossEncoder(self.reranker_model_name)
            except Exception as e:
                logger.warning("Failed to initialize CrossEncoder (%s). Falling back to RRF.", e)
                self.reranker = None
        return self.reranker

    def retrieve_evidence(
        self,
        query: str,
        soc_code: Optional[str] = None,
        major_group: Optional[str] = None,
        aspect: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Executes 3-Stage Hierarchical Aspect-Based Retrieval:
        1. Pre-filtering: narrows corpus to relevant namespace (O(1)).
        2. Hybrid Search: scores sparse BM25 and dense FAISS with RRF fusion.
        3. Cross-Encoder Reranking: scores pair-wise query-passage relevance.
        """
        start_time = time.perf_counter()
        query = query.strip()
        if not query:
            return []

        # ----------------------------------------------------------------------
        # STAGE 1: Pre-filtering Namespace
        # ----------------------------------------------------------------------
        filtered_chunks = self.corpus_manager.filter_chunks(
            major_group=major_group, aspect=aspect, soc_code=soc_code
        )
        allowed_chunk_ids: Optional[Set[str]] = (
            {c["chunk_id"] for c in filtered_chunks} if (major_group or aspect or soc_code) else None
        )

        candidate_count = len(filtered_chunks) if allowed_chunk_ids else self.corpus_manager.total_chunks
        logger.info(
            "RAG Stage 1 Filter: %d candidate chunks (Filters: major_group=%s, aspect=%s, soc=%s)",
            candidate_count,
            major_group,
            aspect,
            soc_code,
        )

        # ----------------------------------------------------------------------
        # STAGE 2: Hybrid Search (BM25 + FAISS Dense)
        # ----------------------------------------------------------------------
        sparse_scores = self.indexer.search_sparse(query, allowed_chunk_ids=allowed_chunk_ids)
        dense_scores = self.indexer.search_dense(query, allowed_chunk_ids=allowed_chunk_ids, top_k=top_k * 3)

        # Combine candidates using Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(sparse_scores.keys()).union(dense_scores.keys())

        # If hybrid searches returned nothing in restricted subset, fallback to all filtered chunks
        if not all_candidate_ids:
            all_candidate_ids = {c["chunk_id"] for c in filtered_chunks[: top_k * 2]}

        # Sort ranks for RRF calculation
        sparse_ranked = sorted(sparse_scores.keys(), key=lambda cid: sparse_scores[cid], reverse=True)
        dense_ranked = sorted(dense_scores.keys(), key=lambda cid: dense_scores[cid], reverse=True)

        sparse_rank_map = {cid: rank + 1 for rank, cid in enumerate(sparse_ranked)}
        dense_rank_map = {cid: rank + 1 for rank, cid in enumerate(dense_ranked)}

        rrf_scores: Dict[str, float] = {}
        k_rrf = 60
        for cid in all_candidate_ids:
            score = 0.0
            if cid in sparse_rank_map:
                score += 0.5 * (1.0 / (k_rrf + sparse_rank_map[cid]))
            if cid in dense_rank_map:
                score += 0.5 * (1.0 / (k_rrf + dense_rank_map[cid]))
            rrf_scores[cid] = score

        # Top candidates for Stage 3 reranking
        top_m_ids = sorted(all_candidate_ids, key=lambda cid: rrf_scores.get(cid, 0.0), reverse=True)[
            : max(top_k * 2, 10)
        ]
        candidates = [self.corpus_manager.get_chunk_by_id(cid) for cid in top_m_ids if cid]
        candidates = [c for c in candidates if c is not None]

        # ----------------------------------------------------------------------
        # STAGE 3: Cross-Encoder Reranking
        # ----------------------------------------------------------------------
        reranker = self._get_reranker()
        if reranker is not None and len(candidates) > 1:
            logger.info("RAG Stage 3: Reranking %d candidates with Cross-Encoder...", len(candidates))
            pairs = [[query, c["content"]] for c in candidates]
            scores = reranker.predict(pairs)
            for idx, c in enumerate(candidates):
                c_copy = dict(c)
                c_copy["relevance_score"] = float(scores[idx])
                candidates[idx] = c_copy
            candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        else:
            for c_idx, c in enumerate(candidates):
                c_copy = dict(c)
                c_copy["relevance_score"] = float(rrf_scores.get(c["chunk_id"], 0.0))
                candidates[c_idx] = c_copy

        final_top_k = candidates[:top_k]
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Evidence retrieval completed in %.2f ms | Returned %d chunks for query: '%s'",
            elapsed_ms,
            len(final_top_k),
            query[:50],
        )

        return final_top_k
