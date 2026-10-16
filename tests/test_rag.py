"""
Unit and Integration Tests for Hierarchical Aspect-Based RAG Engine.
Tests CorpusManager, HybridIndexer, and HierarchicalEvidenceRetriever.
Runs 100% locally and offline without outbound API calls.
"""

from pathlib import Path
import pytest

from source.domain.interfaces.rag import IEvidenceRetriever
from source.model.LLM_model.rag.corpus_manager import CorpusManager
from source.model.LLM_model.rag.indexer import HybridIndexer, simple_tokenize
from source.model.LLM_model.rag.retriever import HierarchicalEvidenceRetriever
from source.infrastructure.config import settings


@pytest.fixture(scope="module")
def corpus_manager() -> CorpusManager:
    """Fixture providing initialized CorpusManager instance."""
    return CorpusManager()


@pytest.fixture(scope="module")
def hybrid_indexer() -> HybridIndexer:
    """Fixture providing loaded HybridIndexer instance."""
    indexer = HybridIndexer()
    indexer.load_index()
    return indexer


@pytest.fixture(scope="module")
def retriever(corpus_manager: CorpusManager, hybrid_indexer: HybridIndexer) -> HierarchicalEvidenceRetriever:
    """Fixture providing HierarchicalEvidenceRetriever with loaded index."""
    return HierarchicalEvidenceRetriever(
        corpus_manager=corpus_manager,
        indexer=hybrid_indexer,
        use_reranker=False,  # Fast RRF testing for unit tests
    )


# ==============================================================================
# 1. Tests for CorpusManager
# ==============================================================================

def test_corpus_manager_initialization_and_metadata(corpus_manager: CorpusManager):
    """Verify loading of all 2,601 micro-chunks across 23 major groups and 3 aspects."""
    assert corpus_manager.total_chunks == 2601
    assert len(corpus_manager._by_major_group) == 23
    assert set(corpus_manager._by_aspect.keys()) == {
        "threat_mechanism",
        "defense_skills",
        "tasks_and_tools",
    }


def test_corpus_manager_filter_major_group(corpus_manager: CorpusManager):
    """Verify Stage 1 pre-filtering strictly isolates chunks by O*NET major group."""
    it_chunks = corpus_manager.filter_chunks(major_group="15")
    assert len(it_chunks) > 0
    assert all(c["major_group"] == "15" for c in it_chunks)
    # Ensure no cross-domain pollution (e.g. healthcare group 29 or legal 23)
    assert not any(c["major_group"] in ("29", "23", "47") for c in it_chunks)


def test_corpus_manager_filter_aspect(corpus_manager: CorpusManager):
    """Verify Stage 1 pre-filtering strictly isolates chunks by aspect."""
    defense_chunks = corpus_manager.filter_chunks(aspect="defense_skills")
    assert len(defense_chunks) == 867
    assert all(c["aspect"] == "defense_skills" for c in defense_chunks)


def test_corpus_manager_filter_combined(corpus_manager: CorpusManager):
    """Verify Stage 1 pre-filtering with compound criteria (group + aspect)."""
    it_defense = corpus_manager.filter_chunks(major_group="15", aspect="defense_skills")
    assert len(it_defense) > 0
    assert all(c["major_group"] == "15" and c["aspect"] == "defense_skills" for c in it_defense)


# ==============================================================================
# 2. Tests for HybridIndexer
# ==============================================================================

def test_simple_tokenize_preserves_tech_tokens():
    """Verify tokenizer correctly handles tech keywords like C++, C#, .NET."""
    tokens = simple_tokenize("Developing microservices with C++, C#, .NET, Python-3, and Docker.")
    assert "cpp" in tokens
    assert "csharp" in tokens
    assert "dotnet" in tokens
    assert "docker" in tokens


def test_hybrid_indexer_sparse_and_dense_search(hybrid_indexer: HybridIndexer):
    """Verify both BM25 sparse search and FAISS dense search return positive scores."""
    assert hybrid_indexer.is_indexed
    assert hybrid_indexer.faiss_index.ntotal == 2601

    # BM25 test
    sparse_res = hybrid_indexer.search_sparse("Python machine learning")
    assert len(sparse_res) > 0
    assert any(score > 0.0 for score in sparse_res.values())

    # FAISS dense test
    dense_res = hybrid_indexer.search_dense("Python machine learning", top_k=10)
    assert len(dense_res) > 0
    assert any(sim > 0.0 for sim in dense_res.values())


# ==============================================================================
# 3. Tests for HierarchicalEvidenceRetriever
# ==============================================================================

def test_retriever_protocol_conformance(retriever: HierarchicalEvidenceRetriever):
    """Verify conformance to IEvidenceRetriever domain protocol."""
    assert isinstance(retriever, IEvidenceRetriever)


def test_retriever_prefiltering_and_aspect_isolation(retriever: HierarchicalEvidenceRetriever):
    """
    Verify retrieval strictly adheres to namespace filters:
    major_group='15' and aspect='defense_skills'.
    """
    query = "Critical thinking and AI defense strategies"
    evidences = retriever.retrieve_evidence(
        query=query,
        major_group="15",
        aspect="defense_skills",
        top_k=3,
    )

    assert len(evidences) == 3
    for ev in evidences:
        assert ev["major_group"] == "15", f"Leaked group {ev['major_group']}"
        assert ev["aspect"] == "defense_skills", f"Leaked aspect {ev['aspect']}"
        assert "content" in ev and len(ev["content"]) > 20
        assert "relevance_score" in ev


def test_retriever_global_fallback(retriever: HierarchicalEvidenceRetriever):
    """Verify global search without namespace filters functions correctly."""
    query = "Statistical data modeling and database management"
    evidences = retriever.retrieve_evidence(query=query, top_k=5)

    assert len(evidences) == 5
    assert all("content" in ev for ev in evidences)


def test_retriever_empty_query(retriever: HierarchicalEvidenceRetriever):
    """Verify empty query gracefully returns an empty list."""
    assert retriever.retrieve_evidence(query="") == []
    assert retriever.retrieve_evidence(query="   ") == []
