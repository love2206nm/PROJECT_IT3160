"""
Agglomerative Clustering Engine Adapter.
Implements IClusteringEngine protocol using Spherical Ward Linkage
on L2-normalized capability vectors to resolve career transition pivots.
"""

from pathlib import Path
import time
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

from source.domain.interfaces.ml_models import IClusteringEngine
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.exceptions import RoleNotFoundError, ModelInferenceError
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class AgglomerativeClusteringEngine(IClusteringEngine):
    """
    Production-grade Career Clustering & Transition Utility Engine.
    Uses Spherical Ward Linkage on unit hypersphere to minimize angular variance
    and eliminate cluster collapse across occupational capability space.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        feature_matrix_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.n_clusters = n_clusters or settings.CLUSTERING_N_CLUSTERS
        self.feature_matrix_path = (
            feature_matrix_path
            or (settings.DATA_CLEAN_DIR / "final" / "soc_feature_matrix.parquet")
        )
        self.metadata_path = (
            metadata_path or (settings.DATA_CLEAN_DIR / "final" / "soc_metadata.parquet")
        )

        self.feature_matrix: Optional[pd.DataFrame] = None
        self.normalized_matrix: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.soc_codes: List[str] = []
        self.soc_to_idx: Dict[str, int] = {}

        self.model: Optional[AgglomerativeClustering] = None
        self.labels_: Optional[np.ndarray] = None
        self.soc_to_cluster: Dict[str, int] = {}
        self.cluster_to_socs: Dict[int, List[str]] = {}

        self.silhouette_: float = 0.0
        self.nmi_: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        feature_matrix: Optional[pd.DataFrame] = None,
        metadata: Optional[pd.DataFrame] = None,
    ) -> "AgglomerativeClusteringEngine":
        """
        Trains Spherical Ward Hierarchical Clustering on L2-normalized capability vectors.
        """
        logger.info("Initializing Agglomerative Clustering Engine (K=%d)...", self.n_clusters)
        start_time = time.perf_counter()

        # 1. Load Data
        if feature_matrix is not None:
            self.feature_matrix = feature_matrix.copy()
        else:
            if not self.feature_matrix_path.exists():
                raise ModelInferenceError(f"Feature matrix not found: '{self.feature_matrix_path}'")
            self.feature_matrix = pd.read_parquet(self.feature_matrix_path)

        if metadata is not None:
            self.metadata = metadata.copy()
        else:
            if not self.metadata_path.exists():
                raise ModelInferenceError(f"Metadata catalog not found: '{self.metadata_path}'")
            self.metadata = pd.read_parquet(self.metadata_path)

        # Align indices
        common_socs = self.feature_matrix.index.intersection(self.metadata.index)
        if len(common_socs) == 0:
            raise ModelInferenceError("Zero common SOC codes found between feature matrix and metadata.")

        self.feature_matrix = self.feature_matrix.loc[common_socs]
        self.metadata = self.metadata.loc[common_socs]

        self.soc_codes = list(self.feature_matrix.index)
        self.feature_names = list(self.feature_matrix.columns)
        self.soc_to_idx = {soc: idx for idx, soc in enumerate(self.soc_codes)}

        # 2. L2-Normalize Feature Vectors (Map to Unit Hypersphere)
        # ||u - v||^2 = 2 * (1 - CosineSim(u, v))
        X_raw = self.feature_matrix.values.astype(np.float32)
        self.normalized_matrix = normalize(X_raw, norm="l2")

        # 3. Fit Agglomerative Clustering with Ward Linkage
        logger.info("Fitting Ward's Minimum Variance Linkage on unit hypersphere...")
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="euclidean",
            linkage="ward",
        )
        self.labels_ = self.model.fit_predict(self.normalized_matrix)

        # 4. Compute Clustering Quality Metrics
        self.silhouette_ = float(silhouette_score(X_raw, self.labels_, metric="cosine"))
        if "major_group" in self.metadata.columns:
            major_groups = self.metadata["major_group"].values
            self.nmi_ = float(normalized_mutual_info_score(major_groups, self.labels_))
        else:
            self.nmi_ = 0.0

        # 5. Build Cluster Lookup Maps
        self.soc_to_cluster.clear()
        self.cluster_to_socs.clear()
        for i in range(self.n_clusters):
            self.cluster_to_socs[i] = []

        for idx, soc in enumerate(self.soc_codes):
            c_label = int(self.labels_[idx])
            self.soc_to_cluster[soc] = c_label
            self.cluster_to_socs[c_label].append(soc)

        self.is_fitted = True
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        cluster_sizes = [len(socs) for socs in self.cluster_to_socs.values()]
        logger.info(
            "Clustering fitted successfully in %.2f ms | Silhouette: %.4f | NMI: %.4f | Min/Max Cluster Size: %d/%d",
            elapsed_ms,
            self.silhouette_,
            self.nmi_,
            min(cluster_sizes),
            max(cluster_sizes),
        )
        return self

    def get_safe_pivots(
        self,
        soc_code: str,
        current_threat: float,
        top_k: int = 5,
        search_cluster_only: bool = True,
    ) -> List[SafeTransitionRole]:
        """
        Finds top safe career pivots ranked by multi-objective Transition Utility.
        """
        if not self.is_fitted:
            self.fit()

        # Normalize SOC code (e.g., '15-1252.00' -> '15-1252')
        soc_clean = soc_code.split(".")[0] if "." in soc_code else soc_code
        if soc_clean not in self.soc_to_idx:
            if soc_code in self.soc_to_idx:
                soc_clean = soc_code
            else:
                logger.warning("SOC code '%s' not present in feature matrix.", soc_code)
                raise RoleNotFoundError(f"Source SOC code '{soc_code}' not found in occupational knowledge base.")

        source_idx = self.soc_to_idx[soc_clean]
        source_vec_norm = self.normalized_matrix[source_idx]
        source_vec_raw = self.feature_matrix.iloc[source_idx].values

        source_meta = self.metadata.loc[soc_clean]
        source_salary = float(source_meta.get("median_salary", 50000.0) or 50000.0)

        # 1. Candidate Target Occupations Selection
        source_cluster = self.soc_to_cluster[soc_clean]
        candidate_socs = [s for s in self.cluster_to_socs[source_cluster] if s != soc_clean]

        # If cluster has too few options, fall back to global market search
        if len(candidate_socs) < top_k and search_cluster_only:
            logger.debug("Cluster %d has only %d peers. Expanding candidate pool globally...", source_cluster, len(candidate_socs))
            candidate_socs = [s for s in self.soc_codes if s != soc_clean]

        # 2. Filter by AI Safety Constraint
        # target_threat must be <= current_threat - safety_delta (or strictly lower if tight)
        delta = settings.SAFETY_MARGIN_DELTA
        valid_candidates = []
        for target_soc in candidate_socs:
            t_meta = self.metadata.loc[target_soc]
            t_threat = float(t_meta.get("threat_score", 50.0))
            if t_threat <= (current_threat - delta):
                valid_candidates.append(target_soc)

        # Fallback if safety constraint with delta filters out all peers: relax delta to strictly < current_threat
        if not valid_candidates:
            logger.debug("No targets found with threat <= current - %.1f. Relaxing to target < current...", delta)
            for target_soc in candidate_socs:
                t_threat = float(self.metadata.loc[target_soc].get("threat_score", 50.0))
                if t_threat < current_threat:
                    valid_candidates.append(target_soc)

        # Final fallback: if candidate's current role is already safest (min threat in cluster), take lowest threat available
        if not valid_candidates:
            logger.debug("Current role has lowest threat in candidate set. Taking top safest peers...")
            sorted_by_threat = sorted(
                candidate_socs,
                key=lambda s: float(self.metadata.loc[s].get("threat_score", 50.0))
            )
            valid_candidates = sorted_by_threat[:top_k * 3]

        # 3. Calculate Multi-Objective Transition Utility
        w_sim = settings.WEIGHT_SIMILARITY
        w_ai = settings.WEIGHT_AI_SAFETY
        w_sal = settings.WEIGHT_SALARY

        scored_targets = []
        for target_soc in valid_candidates:
            t_idx = self.soc_to_idx[target_soc]
            target_vec_norm = self.normalized_matrix[t_idx]
            target_vec_raw = self.feature_matrix.iloc[t_idx].values

            # Cosine similarity between unit normalized vectors is simply dot product
            cos_sim = float(np.dot(source_vec_norm, target_vec_norm))
            cos_sim = max(0.0, min(1.0, cos_sim))

            t_meta = self.metadata.loc[target_soc]
            t_threat = float(t_meta.get("threat_score", 50.0))
            t_salary = float(t_meta.get("median_salary", 50000.0) or 50000.0)
            t_title = str(t_meta.get("soc_title", target_soc))
            t_risk_level = str(t_meta.get("risk_level", "Low"))
            t_defense = list(t_meta.get("defense_skills", [])) if isinstance(t_meta.get("defense_skills"), (list, np.ndarray)) else []

            # AI Safety factor: (1 - Threat/100)
            ai_safety = max(0.0, min(1.0, 1.0 - (t_threat / 100.0)))

            # Salary Advantage factor: normalized relative salary increase
            if source_salary > 0:
                salary_adv = max(0.0, min(1.0, (t_salary - source_salary) / source_salary))
            else:
                salary_adv = 0.5

            # Transition Utility
            utility = (w_sim * cos_sim) + (w_ai * ai_safety) + (w_sal * salary_adv)

            scored_targets.append({
                "soc_code": target_soc,
                "title": t_title,
                "threat_score": t_threat,
                "similarity_score": cos_sim,
                "utility_score": utility,
                "salary_advantage": salary_adv,
                "median_salary": t_salary,
                "risk_level": t_risk_level,
                "defense_skills": t_defense,
                "source_vec": source_vec_raw,
                "target_vec": target_vec_raw,
            })

        # 4. Sort by Utility descending and take Top K
        scored_targets.sort(key=lambda x: x["utility_score"], reverse=True)
        top_targets = scored_targets[:top_k]

        # 5. Build SafeTransitionRole domain entities with SkillGap
        results: List[SafeTransitionRole] = []
        for item in top_targets:
            skill_gap = self._calculate_skill_gap(
                source_vec=item["source_vec"],
                target_vec=item["target_vec"],
                target_title=item["title"],
                target_soc=item["soc_code"],
                cos_sim=item["similarity_score"],
                defense_skills=item["defense_skills"],
            )

            utility_100 = round(float(np.clip(item["utility_score"] * 100.0, 0.0, 100.0)), 2)
            similarity_pct = round(float(np.clip(item["similarity_score"] * 100.0, 0.0, 100.0)), 2)
            threat_reduction = round(float(current_threat - item["threat_score"]), 2)

            role = SafeTransitionRole(
                target_soc_code=item["soc_code"],
                target_title=item["title"],
                utility_score=utility_100,
                similarity_pct=similarity_pct,
                threat_reduction=threat_reduction,
                salary_advantage=round(float(item["salary_advantage"]), 4),
                target_threat_score=round(float(item["threat_score"]), 2),
                target_risk_level=item["risk_level"],
                median_salary=item["median_salary"],
                defense_skills=item["defense_skills"],
                matched_skills=skill_gap.strengths,
                missing_critical_skills=skill_gap.missing_skills,
            )
            results.append(role)

        logger.info(
            "Found %d safe career transition pivots for '%s' (Top 1: '%s', Utility: %.2f/100)",
            len(results),
            soc_code,
            results[0].target_title if results else "None",
            results[0].utility_score if results else 0.0,
        )
        return results

    def _calculate_skill_gap(
        self,
        source_vec: np.ndarray,
        target_vec: np.ndarray,
        target_title: str = "",
        target_soc: str = "",
        cos_sim: float = 0.5,
        defense_skills: Optional[List[str]] = None,
    ) -> SkillGap:
        """
        Calculates granular skill gaps and transferable skills between source and target capability vectors.
        """
        diff = target_vec - source_vec  # positive means target requires more than source

        missing_skills: List[str] = []
        transferable_skills: List[str] = []
        gap_magnitudes: List[float] = []

        for idx, feat_name in enumerate(self.feature_names):
            t_val = target_vec[idx]
            s_val = source_vec[idx]
            d_val = diff[idx]

            # High demand in target and significantly higher than source
            if t_val >= 0.50 and d_val > 0.10:
                missing_skills.append(feat_name)
                gap_magnitudes.append(float(d_val))
            # Both roles demand high capability
            elif s_val >= 0.50 and t_val >= 0.45:
                transferable_skills.append(feat_name)

        # Sort missing skills by largest gap
        if missing_skills and gap_magnitudes:
            sorted_missing = [
                skill for _, skill in sorted(zip(gap_magnitudes, missing_skills), reverse=True)
            ]
            missing_skills = sorted_missing[:8]

        overlap_pct = round(float(np.clip(cos_sim * 100.0, 0.0, 100.0)), 2)

        return SkillGap(
            target_role_title=target_title,
            target_soc_code=target_soc,
            strengths=transferable_skills[:10],
            missing_skills=missing_skills,
            defense_skills=defense_skills or [],
            overlap_percentage=overlap_pct,
        )

    def save_model(self, file_path: Optional[Path] = None) -> Path:
        """Serializes clustering model and lookup dictionaries to joblib artifact."""
        if not self.is_fitted:
            raise ModelInferenceError("Cannot save unfitted clustering engine.")

        path = file_path or (settings.MODEL_DIR / "clustering_model.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_clusters": self.n_clusters,
            "feature_names": self.feature_names,
            "soc_codes": self.soc_codes,
            "soc_to_idx": self.soc_to_idx,
            "normalized_matrix": self.normalized_matrix,
            "labels_": self.labels_,
            "soc_to_cluster": self.soc_to_cluster,
            "cluster_to_socs": self.cluster_to_socs,
            "silhouette_": self.silhouette_,
            "nmi_": self.nmi_,
        }
        joblib.dump(payload, path)
        logger.info("Clustering engine serialized successfully to '%s'.", path)
        return path

    def load_model(
        self,
        file_path: Optional[Path] = None,
        metadata: Optional[pd.DataFrame] = None,
        feature_matrix: Optional[pd.DataFrame] = None,
    ) -> "AgglomerativeClusteringEngine":
        """Deserializes clustering model and lookup maps from joblib artifact."""
        path = file_path or (settings.MODEL_DIR / "clustering_model.joblib")
        if not path.exists():
            raise ModelInferenceError(f"Clustering model artifact not found at '{path}'.")

        payload = joblib.load(path)
        self.n_clusters = payload["n_clusters"]
        self.feature_names = payload["feature_names"]
        self.soc_codes = payload["soc_codes"]
        self.soc_to_idx = payload["soc_to_idx"]
        self.normalized_matrix = payload["normalized_matrix"]
        self.labels_ = payload["labels_"]
        self.soc_to_cluster = payload["soc_to_cluster"]
        self.cluster_to_socs = payload["cluster_to_socs"]
        self.silhouette_ = payload["silhouette_"]
        self.nmi_ = payload["nmi_"]

        # Load metadata and feature_matrix: prefer in-memory, fallback to disk
        if metadata is not None:
            self.metadata = metadata.loc[self.soc_codes]
        elif self.metadata_path and self.metadata_path.exists():
            self.metadata = pd.read_parquet(self.metadata_path).loc[self.soc_codes]

        if feature_matrix is not None:
            self.feature_matrix = feature_matrix.loc[self.soc_codes]
        elif self.feature_matrix_path and self.feature_matrix_path.exists():
            self.feature_matrix = pd.read_parquet(self.feature_matrix_path).loc[self.soc_codes]

        self.is_fitted = True
        logger.info(
            "Clustering engine loaded from '%s' (K=%d, %d SOCs, Silhouette: %.4f).",
            path.name,
            self.n_clusters,
            len(self.soc_codes),
            self.silhouette_,
        )
        return self
