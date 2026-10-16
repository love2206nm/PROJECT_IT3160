"""
O*NET Raw Data Loader & Preprocessor.
Implements IDataLoader protocol to load, clean, and harmonize 9 O*NET datasets.
Fully decouples paths and filenames via RawDataConfig dependency injection.
"""

from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

from source.domain.interfaces.data_pipeline import IDataLoader
from source.data.config import RawDataConfig
from source.data.constants import (
    SURVEY_NOISE_COLUMNS,
    SOFTWARE_TAXONOMY_MAP,
)
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)



class OnetDataLoader:
    """
    Stateful Data Loader and Preprocessor for 9 raw O*NET datasets.
    Accepts RawDataConfig for flexible dependency injection (no hardcoded file paths).
    Harmonizes 8-digit O*NET SOC codes to 6-digit Base SOC codes and drops survey noise.
    """

    def __init__(
        self,
        config: Optional[RawDataConfig] = None,
        raw_dir: Optional[str | Path] = None,
    ) -> None:
        if config is not None:
            self.config = config
        elif raw_dir is not None:
            self.config = RawDataConfig(raw_dir=Path(raw_dir))
        else:
            self.config = RawDataConfig()

        self.raw_dir = Path(self.config.raw_dir)

        self.crosswalk_df_: Optional[pd.DataFrame] = None
        self.skills_df_: Optional[pd.DataFrame] = None
        self.knowledge_df_: Optional[pd.DataFrame] = None
        self.abilities_df_: Optional[pd.DataFrame] = None
        self.software_df_: Optional[pd.DataFrame] = None
        self.software_raw_df_: Optional[pd.DataFrame] = None
        self.tasks_df_: Optional[pd.DataFrame] = None
        self.tasks_raw_df_: Optional[pd.DataFrame] = None
        self.salaries_df_: Optional[pd.DataFrame] = None
        self.threat_df_: Optional[pd.DataFrame] = None
        self.job_titles_df_: Optional[pd.DataFrame] = None
        self.loaded_soc_count_: int = 0
        self.base_soc_list_: List[str] = []

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load and clean all raw datasets, returning a dictionary of harmonized DataFrames.
        """
        logger.info("Starting batch load and harmonization of 9 O*NET raw datasets from '%s'...", self.raw_dir)
        self.load_crosswalk()
        self.load_skills()
        self.load_knowledge()
        self.load_abilities()
        self.load_software_skills()
        self.load_tasks()
        self.load_salaries()
        self.load_threat_index()
        self.load_job_titles()

        self.loaded_soc_count_ = len(self.base_soc_list_)
        logger.info(
            "Successfully loaded and harmonized all 9 O*NET datasets across %d Base SOC codes.",
            self.loaded_soc_count_,
        )

        return {
            "crosswalk": self.crosswalk_df_,
            "skills": self.skills_df_,
            "knowledge": self.knowledge_df_,
            "abilities": self.abilities_df_,
            "software": self.software_df_,
            "tasks": self.tasks_df_,
            "salaries": self.salaries_df_,
            "threat": self.threat_df_,
            "job_titles": self.job_titles_df_,
        }

    def load_crosswalk(self) -> pd.DataFrame:
        """Loads crosswalk dataset to establish the 867 official Base SOC codes index."""
        path = self.config.crosswalk_path
        logger.debug("Loading crosswalk dataset from '%s'...", path)
        df = pd.read_csv(path)
        df["soc_code"] = df["soc_code"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["soc_code"]).set_index("soc_code")
        self.crosswalk_df_ = df
        self.base_soc_list_ = sorted(df.index.tolist())
        logger.info("Loaded crosswalk: established %d unique Base SOC codes index.", len(self.base_soc_list_))
        return self.crosswalk_df_

    def _process_im_lv_dataset(self, file_path: str | Path) -> pd.DataFrame:
        """
        Generic helper for skills, knowledge, and abilities:
        1. Extracts Base SOC (code.split('.')[0]).
        2. Drops survey noise columns.
        3. Computes E = (IM * LV) / 35.0.
        4. Averages across sub-occupations and reindexes to all 867 Base SOCs.
        """
        path = Path(file_path)
        df = pd.read_csv(path)

        # Extract 6-digit Base SOC
        df["soc_code"] = df["O*NET-SOC Code"].astype(str).str.split(".").str[0].str.strip()

        # Drop noise
        drop_cols = [c for c in SURVEY_NOISE_COLUMNS if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Filter IM and LV scales
        df = df[df["Scale ID"].isin(["IM", "LV"])].copy()

        # Pivot to get IM and LV side-by-side for each (O*NET-SOC Code, Element Name)
        pivot = df.pivot_table(
            index=["O*NET-SOC Code", "soc_code", "Element Name"],
            columns="Scale ID",
            values="Data Value",
            aggfunc="mean",
        ).reset_index()

        # Default fill if missing scale
        if "IM" not in pivot.columns:
            pivot["IM"] = 1.0
        if "LV" not in pivot.columns:
            pivot["LV"] = 0.0

        # Calculate Combined Capability Metric E = (IM * LV) / 35.0 in [0, 1]
        pivot["E"] = (pivot["IM"].fillna(1.0) * pivot["LV"].fillna(0.0)) / 35.0
        pivot["E"] = pivot["E"].clip(0.0, 1.0)

        # Group by Base SOC and Element Name (mean across specialty branches)
        base_agg = pivot.groupby(["soc_code", "Element Name"])["E"].mean().unstack(fill_value=0.0)

        # Reindex to full 867 Base SOC list
        if self.base_soc_list_:
            base_agg = base_agg.reindex(self.base_soc_list_, fill_value=0.0)

        result = base_agg.fillna(0.0)
        logger.debug("Processed IM/LV dataset '%s': %d features across %d SOCs.", path.name, result.shape[1], result.shape[0])
        return result

    def load_skills(self) -> pd.DataFrame:
        """Loads and processes basic skills dataset."""
        logger.debug("Loading basic skills dataset from '%s'...", self.config.skills_path)
        self.skills_df_ = self._process_im_lv_dataset(self.config.skills_path)
        logger.info("Loaded skills: %d indicators.", self.skills_df_.shape[1])
        return self.skills_df_

    def load_knowledge(self) -> pd.DataFrame:
        """Loads and processes knowledge domains dataset."""
        logger.debug("Loading knowledge domains dataset from '%s'...", self.config.knowledge_path)
        self.knowledge_df_ = self._process_im_lv_dataset(self.config.knowledge_path)
        logger.info("Loaded knowledge domains: %d indicators.", self.knowledge_df_.shape[1])
        return self.knowledge_df_

    def load_abilities(self) -> pd.DataFrame:
        """Loads and processes abilities dataset."""
        logger.debug("Loading abilities dataset from '%s'...", self.config.abilities_path)
        self.abilities_df_ = self._process_im_lv_dataset(self.config.abilities_path)
        logger.info("Loaded abilities: %d indicators.", self.abilities_df_.shape[1])
        return self.abilities_df_

    def load_software_skills(self) -> pd.DataFrame:
        """
        Loads software skills dataset:
        - Harmonizes Base SOC.
        - Categorizes tools into 20 Software Taxonomy Categories.
        - Computes category prevalence scores for each SOC in [0, 1].
        - Preserves catalog metadata.
        """
        path = self.config.software_skills_path
        logger.debug("Loading software skills catalog from '%s'...", path)
        df = pd.read_csv(path)

        df["soc_code"] = df["O*NET-SOC Code"].astype(str).str.split(".").str[0].str.strip()
        df["tool_name"] = df["Workplace Example"].astype(str).str.strip()
        df["element_name"] = df["Element Name"].astype(str).str.strip().str.lower()
        df["is_hot_tech"] = df["Hot Technology"].astype(str).str.upper().isin(["Y", "TRUE", "1"])
        df["is_in_demand"] = df["In Demand"].astype(str).str.upper().isin(["Y", "TRUE", "1"])

        # Map element_name to 20 taxonomy categories
        def map_taxonomy(elem: str) -> str:
            elem_lower = elem.lower()
            for cat, keywords in SOFTWARE_TAXONOMY_MAP.items():
                if any(kw in elem_lower for kw in keywords):
                    return cat
            return "tech_cat_office_spreadsheets"

        df["taxonomy_category"] = df["element_name"].apply(map_taxonomy)

        # Compute category occurrence count per SOC
        cat_counts = df.groupby(["soc_code", "taxonomy_category"])["tool_name"].count().unstack(fill_value=0.0)

        # Normalize by maximum category count per SOC to get [0, 1] scale
        row_max = cat_counts.max(axis=1).replace(0, 1.0)
        cat_normalized = cat_counts.div(row_max, axis=0).clip(0.0, 1.0)

        # Ensure all 20 categories exist as columns
        for cat in SOFTWARE_TAXONOMY_MAP.keys():
            if cat not in cat_normalized.columns:
                cat_normalized[cat] = 0.0

        cat_normalized = cat_normalized[list(SOFTWARE_TAXONOMY_MAP.keys())]

        if self.base_soc_list_:
            cat_normalized = cat_normalized.reindex(self.base_soc_list_, fill_value=0.0)

        self.software_df_ = cat_normalized.fillna(0.0)
        self.software_raw_df_ = df
        logger.info(
            "Loaded software skills: %d unique tools classified across 20 taxonomy categories.",
            len(df["tool_name"].unique()),
        )
        return self.software_df_

    def load_tasks(self) -> pd.DataFrame:
        """
        Loads task ratings dataset:
        Computes complexity levels (Data, People, Things) and skill breadth per Base SOC.
        """
        path = self.config.task_ratings_path
        logger.debug("Loading task ratings from '%s'...", path)
        df = pd.read_csv(path)

        df["soc_code"] = df["O*NET-SOC Code"].astype(str).str.split(".").str[0].str.strip()

        im_df = df[df["Scale ID"] == "IM"].copy()
        im_agg = im_df.groupby("soc_code")["Data Value"].agg(
            task_count="count",
            mean_importance="mean",
            high_importance_count=lambda x: (x >= 3.5).sum(),
        )

        if self.base_soc_list_:
            im_agg = im_agg.reindex(self.base_soc_list_, fill_value=0.0)

        self.tasks_df_ = im_agg.fillna(0.0)
        self.tasks_raw_df_ = df
        logger.info("Loaded tasks: %d total rated tasks processed.", len(df))
        return self.tasks_df_

    def load_salaries(self) -> pd.DataFrame:
        """
        Loads real pay salary dataset:
        Aggregates median real_p50 salary across all metro areas for each Base SOC.
        """
        path = self.config.salaries_path
        logger.debug("Loading real pay salaries from '%s'...", path)
        df = pd.read_csv(path, low_memory=False)

        df["soc_code"] = df["soc_code"].astype(str).str.strip()
        df["real_p50"] = pd.to_numeric(df["real_p50"], errors="coerce")

        sal_agg = df.groupby("soc_code")["real_p50"].median().to_frame(name="median_real_p50")

        if self.base_soc_list_:
            sal_agg = sal_agg.reindex(self.base_soc_list_)

        self.salaries_df_ = sal_agg
        valid_salaries = sal_agg["median_real_p50"].dropna().count()
        logger.info("Loaded salaries: %d Base SOCs with empirical metro salary data.", valid_salaries)
        return self.salaries_df_

    def load_threat_index(self) -> pd.DataFrame:
        """
        Loads career threat index dataset:
        Provides threat_score, sub-scores, salary bounds, defense skills, and insights.
        """
        path = self.config.threat_index_path
        logger.debug("Loading career threat index from '%s'...", path)
        df = pd.read_csv(path)

        df["soc_code"] = df["soc_code"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["soc_code"]).set_index("soc_code")

        if self.base_soc_list_:
            df = df.reindex(self.base_soc_list_)

        self.threat_df_ = df
        matched_threat = df["score"].dropna().count() if "score" in df.columns else len(df.dropna())
        logger.info("Loaded threat index: %d Base SOCs with direct threat records.", matched_threat)
        return self.threat_df_

    def load_job_titles(self) -> pd.DataFrame:
        """Loads job titles dataset (54k market titles) for rapid entity matching."""
        path = self.config.job_titles_path
        logger.debug("Loading market job titles from '%s'...", path)
        df = pd.read_csv(path)
        df["soc_code"] = df["O*NET-SOC Code"].astype(str).str.split(".").str[0].str.strip()
        df["job_title_clean"] = df["Job Title"].astype(str).str.strip()
        self.job_titles_df_ = df[["soc_code", "job_title_clean", "Title"]].dropna()
        logger.info("Loaded job titles: %d clean market title mappings.", len(self.job_titles_df_))
        return self.job_titles_df_
