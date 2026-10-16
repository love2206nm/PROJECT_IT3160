"""
Data Pipeline Domain Protocols (Interfaces).
Defines abstract contracts for data ingestion, feature transformation, and metadata/RAG generation.
"""

from typing import Protocol, Dict, Any, List, Tuple, runtime_checkable
import pandas as pd


@runtime_checkable
class IDataLoader(Protocol):
    """Protocol for loading and initial cleaning of raw O*NET datasets."""
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load and clean all raw O*NET data files, harmonized to Base SOC index."""
        ...


@runtime_checkable
class IFeatureTransformer(Protocol):
    """Protocol for Two-Stage Feature Pipeline (120D Bronze -> ~55D Gold Store)."""
    def extract_120d(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Stage 1: Extract zero-leakage 120-dimensional feature space."""
        ...

    def prune_features(self, full_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Stage 2: Perform Group-wise Correlation Pruning (|r| >= threshold) and cross-group tie-break."""
        ...

    def fit_transform(
        self, raw_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Execute Stage 1 and Stage 2 end-to-end, returning (full_matrix, gold_matrix, report)."""
        ...


@runtime_checkable
class IMetadataBuilder(Protocol):
    """Protocol for building Metadata Catalog and Hierarchical RAG Corpus."""
    def build_metadata(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build consolidated metadata DataFrame with imputed salary, benchmark threat, and software catalog."""
        ...

    def build_rag_corpus(
        self, metadata_df: pd.DataFrame, raw_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Generate 2,601 aspect-based micro-chunks tagged with major_group namespace for RAG."""
        ...
