"""
Two-Stage Feature Transformer.
Implements IFeatureTransformer protocol:
- Stage 1: 120D Zero-Leakage Bronze Feature Store.
- Stage 2: Group-wise Correlation Pruning (|r| >= 0.85) Gold Store.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from source.domain.interfaces.data_pipeline import IFeatureTransformer
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)



class TwoStageFeatureTransformer:
    """
    Stateful Two-Stage Feature Transformer:
    - Stage 1: Extracts 120-dimensional capability space (Zero Threat Score Leakage).
    - Stage 2: Performs Group-wise Correlation Pruning to eliminate multicollinearity.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        cross_group_threshold: float = 0.95,
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.cross_group_threshold = cross_group_threshold

        self.full_feature_names_: List[str] = []
        self.pruned_feature_names_: List[str] = []
        self.feature_groups_: Dict[str, List[str]] = {}
        self.dropped_features_by_group_: Dict[str, List[str]] = {}
        self.pruning_report_: Dict[str, Any] = {}

        self.feature_matrix_full_: Optional[pd.DataFrame] = None
        self.feature_matrix_gold_: Optional[pd.DataFrame] = None

    def _clean_column_name(self, prefix: str, name: str) -> str:
        """Sanitizes text to a clean snake_case column identifier."""
        clean = (
            name.lower()
            .replace(" - ", "_")
            .replace("-", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace(",", "")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
        )
        while "__" in clean:
            clean = clean.replace("__", "_")
        return f"{prefix}_{clean.strip('_')}"

    def extract_120d(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Stage 1: Assembles the 120-dimensional Zero-Leakage feature matrix.
        - Group 1: 10 Basic Skills
        - Group 2: 33 Knowledge Domains
        - Group 3: 52 Abilities
        - Group 4: 20 Software Taxonomy Categories
        - Group 5: 5 Economic & Complexity Indicators
        """
        logger.info("Stage 1: Assembling 120-dimensional Zero-Leakage capability feature space...")
        skills_df = raw_data["skills"].copy()
        knowledge_df = raw_data["knowledge"].copy()
        abilities_df = raw_data["abilities"].copy()
        software_df = raw_data["software"].copy()
        tasks_df = raw_data["tasks"].copy()
        salaries_df = raw_data["salaries"].copy()
        threat_df = raw_data["threat"].copy()
        crosswalk_df = raw_data["crosswalk"].copy()

        base_index = crosswalk_df.index

        # 1. Rename Skills (10 features)
        skill_cols = {col: self._clean_column_name("skill", col) for col in skills_df.columns}
        skills_renamed = skills_df.rename(columns=skill_cols).reindex(base_index, fill_value=0.0)

        # 2. Rename Knowledge (33 features)
        knowledge_cols = {col: self._clean_column_name("knowledge", col) for col in knowledge_df.columns}
        knowledge_renamed = knowledge_df.rename(columns=knowledge_cols).reindex(base_index, fill_value=0.0)

        # 3. Rename Abilities (52 features)
        ability_cols = {col: self._clean_column_name("ability", col) for col in abilities_df.columns}
        abilities_renamed = abilities_df.rename(columns=ability_cols).reindex(base_index, fill_value=0.0)

        # 4. Software Taxonomy (20 features)
        software_renamed = software_df.reindex(base_index, fill_value=0.0)

        # 5. Economic & Complexity Indicators (5 features - ZERO LEAKAGE)
        # 5.1 Normalized Salary: Real Metro median with Threat Index fallback
        sal = salaries_df["median_real_p50"].copy()
        fallback_sal = (threat_df["salary_low_usd"] + threat_df["salary_high_usd"]) / 2.0
        combined_sal = sal.combine_first(fallback_sal).fillna(65000.0)  # National median fallback if missing

        min_s, max_s = combined_sal.min(), combined_sal.max()
        norm_sal = (combined_sal - min_s) / (max_s - min_s + 1e-6)

        # 5.2 Complexity Data Level (cognitive reasoning + info ordering + analytical knowledge)
        data_knowledge = [c for c in knowledge_renamed.columns if any(k in c for k in ["math", "computer", "engineer", "economic"])]
        complexity_data = knowledge_renamed[data_knowledge].mean(axis=1)

        # 5.3 Complexity People Level (communication, active listening, social perception)
        people_cols = [c for c in skills_renamed.columns if any(k in c for k in ["speaking", "listening", "learning", "monitoring"])]
        complexity_people = skills_renamed[people_cols].mean(axis=1)

        # 5.4 Complexity Things Level (mechanical, physical, bodily coordination)
        things_cols = [c for c in abilities_renamed.columns if any(k in c for k in ["manual", "finger", "multilimb", "trunk", "stamina", "strength"])]
        complexity_things = abilities_renamed[things_cols].mean(axis=1) if things_cols else pd.Series(0.0, index=base_index)

        # 5.5 Skill Breadth Index: Software category coverage
        skill_breadth = (software_renamed > 0.0).mean(axis=1)

        econ_df = pd.DataFrame(
            {
                "econ_normalized_salary": norm_sal.clip(0.0, 1.0),
                "complexity_data_level": complexity_data.clip(0.0, 1.0),
                "complexity_people_level": complexity_people.clip(0.0, 1.0),
                "complexity_things_level": complexity_things.clip(0.0, 1.0),
                "skill_breadth_index": skill_breadth.clip(0.0, 1.0),
            },
            index=base_index,
        )

        # Record group mappings
        self.feature_groups_ = {
            "skills": list(skills_renamed.columns),
            "knowledge": list(knowledge_renamed.columns),
            "abilities": list(abilities_renamed.columns),
            "software": list(software_renamed.columns),
            "economic_complexity": list(econ_df.columns),
        }

        # Concatenate 120D space
        matrix_120d = pd.concat(
            [skills_renamed, knowledge_renamed, abilities_renamed, software_renamed, econ_df],
            axis=1,
        ).astype(np.float32)

        # Fill any lingering NaNs with 0.0
        matrix_120d = matrix_120d.fillna(0.0)

        self.full_feature_names_ = list(matrix_120d.columns)
        self.feature_matrix_full_ = matrix_120d
        logger.info(
            "Stage 1 completed: Generated 120D feature matrix with shape %s (0 missing values).",
            matrix_120d.shape,
        )

        return self.feature_matrix_full_

    def prune_features(self, full_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Stage 2: Group-wise Correlation Pruning (|r| >= threshold) and cross-group tie-break.
        Preserves balanced representation across domains.
        """
        logger.info(
            "Stage 2: Starting Group-wise Correlation Pruning (Intra-group |r| >= %.2f, Cross-group |r| >= %.2f)...",
            self.correlation_threshold,
            self.cross_group_threshold,
        )
        kept_features_by_group: Dict[str, List[str]] = {}
        dropped_details: List[Dict[str, Any]] = []

        # 1. Group-wise Pruning within each group
        for group_name in ["skills", "knowledge", "abilities", "software"]:
            cols = self.feature_groups_.get(group_name, [])
            sub_df = full_matrix[cols]
            corr = sub_df.corr().abs()

            dropped_in_group = set()
            for i in range(len(cols)):
                col_i = cols[i]
                if col_i in dropped_in_group:
                    continue
                for j in range(i + 1, len(cols)):
                    col_j = cols[j]
                    if col_j in dropped_in_group:
                        continue
                    r_val = float(corr.loc[col_i, col_j])
                    if r_val >= self.correlation_threshold:
                        # Tie-breaker: keep the feature with higher variance
                        var_i = float(sub_df[col_i].var())
                        var_j = float(sub_df[col_j].var())
                        if var_i >= var_j:
                            dropped = col_j
                            kept = col_i
                        else:
                            dropped = col_i
                            kept = col_j

                        dropped_in_group.add(dropped)
                        dropped_details.append({
                            "group": group_name,
                            "type": "intra_group",
                            "dropped": dropped,
                            "kept": kept,
                            "correlation": round(r_val, 4),
                            "reason": f"Intra-group multicollinearity (|r| = {r_val:.3f} >= {self.correlation_threshold})",
                        })

                        if dropped == col_i:
                            break  # col_i is dropped, move to next

            kept_features_by_group[group_name] = [c for c in cols if c not in dropped_in_group]
            self.dropped_features_by_group_[group_name] = list(dropped_in_group)

        # Economic and complexity indicators are preserved as core business metrics
        kept_features_by_group["economic_complexity"] = list(self.feature_groups_.get("economic_complexity", []))
        self.dropped_features_by_group_["economic_complexity"] = []

        # 2. Cross-group tie-break for near-identical duplicate concepts (|r| >= 0.95)
        # e.g., skill_mathematics vs ability_mathematical_reasoning
        candidate_cols = []
        for g_cols in kept_features_by_group.values():
            candidate_cols.extend(g_cols)

        cross_df = full_matrix[candidate_cols]
        cross_corr = cross_df.corr().abs()
        cross_dropped = set()

        for i in range(len(candidate_cols)):
            col_i = candidate_cols[i]
            if col_i in cross_dropped:
                continue
            for j in range(i + 1, len(candidate_cols)):
                col_j = candidate_cols[j]
                if col_j in cross_dropped:
                    continue
                r_val = float(cross_corr.loc[col_i, col_j])
                if r_val >= self.cross_group_threshold:
                    var_i = float(cross_df[col_i].var())
                    var_j = float(cross_df[col_j].var())

                    # Domain balance rule: If one feature is from skills and the other from abilities,
                    # drop the ability to prevent abilities (52) from wiping out skills (10)
                    g_i = next((g for g, cols in kept_features_by_group.items() if col_i in cols), "")
                    g_j = next((g for g, cols in kept_features_by_group.items() if col_j in cols), "")

                    if g_i == "skills" and g_j == "abilities":
                        dropped = col_j
                        kept = col_i
                    elif g_j == "skills" and g_i == "abilities":
                        dropped = col_i
                        kept = col_j
                    else:
                        dropped = col_j if var_i >= var_j else col_i
                        kept = col_i if var_i >= var_j else col_j

                    cross_dropped.add(dropped)
                    dropped_details.append({
                        "group": "cross_group",
                        "type": "cross_group_tie_break",
                        "dropped": dropped,
                        "kept": kept,
                        "correlation": round(r_val, 4),
                        "reason": f"Cross-group duplication (|r| = {r_val:.3f} >= {self.cross_group_threshold})",
                    })

                    if dropped == col_i:
                        break

        final_cols = [c for c in candidate_cols if c not in cross_dropped]
        gold_matrix = full_matrix[final_cols].copy()

        self.pruned_feature_names_ = final_cols
        self.feature_matrix_gold_ = gold_matrix

        # Build comprehensive pruning report
        self.pruning_report_ = {
            "initial_feature_count": len(self.full_feature_names_),
            "pruned_feature_count": len(final_cols),
            "reduction_percentage": round((1.0 - len(final_cols) / len(self.full_feature_names_)) * 100, 2),
            "correlation_threshold": self.correlation_threshold,
            "cross_group_threshold": self.cross_group_threshold,
            "feature_counts_by_group": {
                g: {
                    "original": len(self.feature_groups_[g]),
                    "kept": len([c for c in final_cols if c in self.feature_groups_[g]]),
                    "dropped": len(self.feature_groups_[g]) - len([c for c in final_cols if c in self.feature_groups_[g]]),
                }
                for g in self.feature_groups_
            },
            "dropped_details": dropped_details,
        }

        logger.info(
            "Stage 2 completed: Pruned %d initial features down to %d model-ready gold features (%.2f%% reduction).",
            len(self.full_feature_names_),
            len(final_cols),
            self.pruning_report_["reduction_percentage"],
        )

        return self.feature_matrix_gold_, self.pruning_report_

    def fit_transform(
        self, raw_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Executes Stage 1 and Stage 2 end-to-end."""
        logger.info("Executing TwoStageFeatureTransformer fit_transform pipeline...")
        full_matrix = self.extract_120d(raw_data)
        gold_matrix, report = self.prune_features(full_matrix)
        logger.info("TwoStageFeatureTransformer fit_transform pipeline finished successfully.")
        return full_matrix, gold_matrix, report
