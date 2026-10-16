"""
O*NET Hierarchical Aspect-Based Corpus Manager.
Loads, validates, and provides O(1) in-memory namespace pre-filtering across 2,601 micro-chunks.
"""

from pathlib import Path
import json
from typing import Dict, Any, List, Optional, Set
from source.infrastructure.config import settings
from source.domain.interfaces.rag import IKnowledgeCorpus
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class CorpusManager(IKnowledgeCorpus):
    """
    Manages the 2,601 Hierarchical Aspect-Based Micro-Chunks generated from O*NET.
    Implements IKnowledgeCorpus domain protocol with O(1) in-memory pre-filtering.
    """

    def __init__(self, corpus_path: Optional[Path] = None) -> None:
        self.corpus_path = (
            corpus_path or (settings.DATA_CLEAN_DIR / "final" / "rag_corpus.json")
        )
        self._chunks: List[Dict[str, Any]] = []
        self._chunk_by_id: Dict[str, Dict[str, Any]] = {}
        self._by_major_group: Dict[str, List[int]] = {}
        self._by_aspect: Dict[str, List[int]] = {}
        self._by_soc_code: Dict[str, List[int]] = {}
        self._is_loaded: bool = False

        self.load_corpus()

    def load_corpus(self) -> None:
        """Loads and indexes the JSON corpus into in-memory lookup structures."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"RAG Corpus file not found at: '{self.corpus_path}'. "
                "Ensure data pipeline metadata builder has been executed."
            )

        logger.info("Loading RAG Knowledge Corpus from '%s'...", self.corpus_path)
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)

        # Clear and build indices
        self._chunk_by_id.clear()
        self._by_major_group.clear()
        self._by_aspect.clear()
        self._by_soc_code.clear()

        for idx, chunk in enumerate(self._chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            soc_code = str(chunk.get("soc_code", "")).strip()
            major_group = str(chunk.get("major_group", "")).strip()
            aspect = str(chunk.get("aspect", "")).strip()

            self._chunk_by_id[chunk_id] = chunk

            # Index by major_group
            if major_group:
                self._by_major_group.setdefault(major_group, []).append(idx)

            # Index by aspect
            if aspect:
                self._by_aspect.setdefault(aspect, []).append(idx)

            # Index by soc_code
            if soc_code:
                self._by_soc_code.setdefault(soc_code, []).append(idx)
                # Also index base soc (without .00) if applicable
                if "." in soc_code:
                    base_soc = soc_code.split(".")[0]
                    self._by_soc_code.setdefault(base_soc, []).append(idx)

        self._is_loaded = True
        logger.info(
            "RAG Knowledge Corpus loaded: %d chunks across %d major groups and %d aspects.",
            len(self._chunks),
            len(self._by_major_group),
            len(self._by_aspect),
        )

    def filter_chunks(
        self,
        major_group: Optional[str] = None,
        aspect: Optional[str] = None,
        soc_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage 1 Pre-filtering: Filters chunks matching all specified criteria.
        If no filters are provided, returns all chunks (global fallback).

        Args:
            major_group: Optional 2-digit O*NET major occupational group (e.g., '15')
            aspect: Optional aspect ('threat_mechanism', 'defense_skills', 'tasks_and_tools')
            soc_code: Optional SOC code (e.g., '15-1252' or '15-1252.00')

        Returns:
            List[Dict[str, Any]]: Filtered subset of micro-chunks
        """
        if not self._is_loaded:
            self.load_corpus()

        candidate_sets: List[Set[int]] = []

        if major_group:
            mg = str(major_group).strip()
            indices = set(self._by_major_group.get(mg, []))
            candidate_sets.append(indices)

        if aspect:
            asp = str(aspect).strip()
            indices = set(self._by_aspect.get(asp, []))
            candidate_sets.append(indices)

        if soc_code:
            soc = str(soc_code).strip()
            indices = set(self._by_soc_code.get(soc, []))
            candidate_sets.append(indices)

        if not candidate_sets:
            return list(self._chunks)

        # Intersection of all active filter sets
        final_indices = set.intersection(*candidate_sets)
        return [self._chunks[i] for i in sorted(final_indices)]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """O(1) retrieval by unique chunk identifier."""
        return self._chunk_by_id.get(chunk_id)

    @property
    def all_chunks(self) -> List[Dict[str, Any]]:
        """Returns the full list of 2,601 chunks."""
        return self._chunks

    @property
    def total_chunks(self) -> int:
        """Returns total chunk count."""
        return len(self._chunks)
