"""
RAG Engine Domain Protocols (Ports).
Defines abstract contracts for factual evidence retrieval from O*NET knowledge corpus.
"""

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable


@runtime_checkable
class IEvidenceRetriever(Protocol):
    """Protocol for retrieving aspect-grounded evidence chunks from O*NET RAG corpus."""

    def retrieve_evidence(
        self,
        query: str,
        soc_code: Optional[str] = None,
        major_group: Optional[str] = None,
        aspect: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top factual micro-chunks relevant to candidate transition or threat mechanism.

        Args:
            query: Natural language search query or candidate description
            soc_code: Optional SOC code filter
            major_group: Optional 2-digit major occupation filter for pre-filtering
            aspect: Optional aspect filter ('threat_mechanism', 'defense_skills', 'tasks_and_tools')
            top_k: Number of chunks to retrieve

        Returns:
            List[Dict[str, Any]]: List of retrieved micro-chunk records with content and metadata
        """
        ...


@runtime_checkable
class IKnowledgeCorpus(Protocol):
    """Protocol for in-memory O*NET knowledge corpus supporting namespace pre-filtering."""

    @property
    def all_chunks(self) -> List[Dict[str, Any]]:
        """List of all available micro-chunks in the corpus."""
        ...

    @property
    def total_chunks(self) -> int:
        """Total number of micro-chunks in the corpus."""
        ...

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by its unique ID."""
        ...

    def filter_chunks(
        self,
        major_group: Optional[str] = None,
        aspect: Optional[str] = None,
        soc_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """O(1) in-memory namespace pre-filtering."""
        ...
