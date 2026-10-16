"""
O*NET Metadata Catalog & Hierarchical Aspect RAG Corpus Builder.
Implements IMetadataBuilder protocol.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from source.domain.interfaces.data_pipeline import IMetadataBuilder
from source.data.constants import UNMATCHED_SOC_BENCHMARKS, ONET_MAJOR_GROUPS
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)



class OnetMetadataBuilder:
    """
    Builds the consolidated Metadata Catalog (soc_metadata.parquet) and
    the 2,601 Hierarchical Aspect-Based Micro-Chunks (rag_corpus.json).
    """

    def __init__(self) -> None:
        self.metadata_df_: Optional[pd.DataFrame] = None
        self.rag_corpus_: Optional[List[Dict[str, Any]]] = None
        self.imputed_salary_count_: int = 0
        self.benchmark_risk_count_: int = 0

    def build_metadata(
        self, raw_data: Dict[str, pd.DataFrame], raw_loader: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Builds the complete 867-row metadata DataFrame with 100% salary coverage,
        risk profiles, 3 defense skills, and 8,753 software tools catalog.
        """
        logger.info("Building comprehensive O*NET Metadata Catalog...")
        crosswalk = raw_data["crosswalk"].copy()
        threat = raw_data["threat"].copy()
        salaries = raw_data["salaries"].copy()

        # Build lookup table by slug from threat index
        if "slug" in threat.columns:
            threat_by_slug = threat.dropna(subset=["slug"]).drop_duplicates(subset=["slug"]).set_index("slug")
        else:
            threat_by_slug = pd.DataFrame()

        # Software raw records
        software_raw = getattr(raw_loader, "software_raw_df_", None)
        tasks_raw = getattr(raw_loader, "tasks_raw_df_", None)

        metadata_records = []
        imputed_sal_count = 0
        benchmark_count = 0

        for soc_code, cw_row in crosswalk.iterrows():
            soc_code_str = str(soc_code)
            soc_title = cw_row.get("soc_title", "")
            major_prefix = soc_code_str.split("-")[0]
            role_slug = cw_row.get("nearest_role_slug")

            # 1. Threat and Risk Profile Lookup (First by slug, fallback by soc_code)
            threat_row = None
            if pd.notna(role_slug) and role_slug in threat_by_slug.index:
                threat_row = threat_by_slug.loc[role_slug]
            elif soc_code in threat.index:
                candidate = threat.loc[soc_code]
                if isinstance(candidate, pd.DataFrame):
                    threat_row = candidate.iloc[0]
                else:
                    threat_row = candidate

            is_threat_missing = (
                threat_row is None
                or pd.isna(threat_row.get("score"))
                or pd.isna(threat_row.get("risk_level"))
            )

            if not is_threat_missing:
                t_score = float(threat_row.get("score", 50.0))
                r_level = str(threat_row.get("risk_level", "Moderate"))
                sub_auto = float(threat_row.get("sub_task_automation", 50.0))
                sub_tool = float(threat_row.get("sub_tool_maturity", 50.0))
                sub_agentic = float(threat_row.get("sub_agentic_exposure", 50.0))
                d1 = str(threat_row.get("defense_skill_1", "Problem Solving"))
                d2 = str(threat_row.get("defense_skill_2", "Critical Thinking"))
                d3 = str(threat_row.get("defense_skill_3", "Adaptive Learning"))
                insight = str(threat_row.get("insight", "Standard O*NET occupational profile."))
                sal_low = float(threat_row.get("salary_low_usd", 45000.0)) if not pd.isna(threat_row.get("salary_low_usd")) else 45000.0
                sal_high = float(threat_row.get("salary_high_usd", 85000.0)) if not pd.isna(threat_row.get("salary_high_usd")) else 85000.0
            else:
                # Benchmark fallback for unmatched SOCs
                benchmark_count += 1
                bm = UNMATCHED_SOC_BENCHMARKS.get(major_prefix, UNMATCHED_SOC_BENCHMARKS["45"])
                t_score = bm["threat_score"]
                r_level = bm["risk_level"]
                sub_auto = bm["sub_task_automation"]
                sub_tool = bm["sub_tool_maturity"]
                sub_agentic = bm["sub_agentic_exposure"]
                d1, d2, d3 = bm["defense_skills"]
                insight = bm["insight"]
                sal_low, sal_high = 38000.0, 68000.0

            defense_skills = [d1, d2, d3]

            # 2. Median Salary (Metro Pay with fallback)
            metro_sal = salaries.loc[soc_code, "median_real_p50"] if soc_code in salaries.index else np.nan
            if not pd.isna(metro_sal):
                final_salary = float(metro_sal)
            else:
                imputed_sal_count += 1
                final_salary = float((sal_low + sal_high) / 2.0)

            # 3. Software Tools Catalog
            soc_tools = []
            hot_tech_list = []
            if software_raw is not None:
                tool_sub = software_raw[software_raw["soc_code"] == soc_code_str]
                for _, t_row in tool_sub.iterrows():
                    t_name = str(t_row["tool_name"])
                    is_hot = bool(t_row["is_hot_tech"])
                    is_dem = bool(t_row["is_in_demand"])
                    cat = str(t_row["taxonomy_category"])
                    soc_tools.append({
                        "name": t_name,
                        "hot_tech": is_hot,
                        "in_demand": is_dem,
                        "category": cat,
                    })
                    if is_hot and t_name not in hot_tech_list:
                        hot_tech_list.append(t_name)

            # 4. Critical Tasks
            critical_tasks = []
            if tasks_raw is not None:
                tasks_sub = tasks_raw[tasks_raw["soc_code"] == soc_code_str]
                im_tasks = tasks_sub[tasks_sub["Scale ID"] == "IM"].sort_values(by="Data Value", ascending=False)
                for _, task_row in im_tasks.head(5).iterrows():
                    critical_tasks.append(str(task_row["Task"]))

            metadata_records.append({
                "soc_code": soc_code_str,
                "soc_title": soc_title,
                "major_group": major_prefix,
                "major_group_name": ONET_MAJOR_GROUPS.get(major_prefix, "General Occupations"),
                "threat_score": round(t_score, 1),
                "risk_level": r_level,
                "sub_task_automation": round(sub_auto, 1),
                "sub_tool_maturity": round(sub_tool, 1),
                "sub_agentic_exposure": round(sub_agentic, 1),
                "defense_skills": defense_skills,
                "insight": insight,
                "median_salary": round(final_salary, 1),
                "software_skills_catalog": soc_tools,
                "top_hot_technologies": hot_tech_list,
                "critical_tasks": critical_tasks,
            })

        meta_df = pd.DataFrame(metadata_records).set_index("soc_code")
        self.metadata_df_ = meta_df
        self.imputed_salary_count_ = imputed_sal_count
        self.benchmark_risk_count_ = benchmark_count
        logger.info(
            "Metadata Catalog built: %d SOCs (Salaries imputed: %d, Benchmark risks: %d).",
            len(meta_df),
            imputed_sal_count,
            benchmark_count,
        )
        return self.metadata_df_

    def build_rag_corpus(
        self, metadata_df: pd.DataFrame, raw_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[Dict[str, Any]]:
        """
        Builds exactly 2,601 Aspect-Based Micro-Chunks (867 x 3 aspects):
        1. threat_mechanism
        2. defense_skills
        3. tasks_and_tools
        Tagged with major_group for fast pre-filtering search.
        """
        logger.info(
            "Building Hierarchical Aspect-Based RAG Corpus (3 micro-chunks per SOC across %d SOCs)...",
            len(metadata_df),
        )
        corpus: List[Dict[str, Any]] = []

        for soc_code, row in metadata_df.iterrows():
            soc_code_str = str(soc_code)
            soc_title = row["soc_title"]
            major_group = row["major_group"]
            t_score = row["threat_score"]
            r_level = row["risk_level"]
            sub_auto = row["sub_task_automation"]
            sub_tool = row["sub_tool_maturity"]
            sub_agentic = row["sub_agentic_exposure"]
            defense_skills = row["defense_skills"]
            insight = row["insight"]
            hot_tech = row["top_hot_technologies"]
            tasks = row["critical_tasks"]

            # Aspect 1: Threat Mechanism
            threat_content = (
                f"Cơ chế đe dọa và rủi ro tự động hóa AI đối với {soc_title} (SOC {soc_code_str}): "
                f"Điểm đe dọa tổng hợp {t_score:.1f}/100 thuộc nhóm rủi ro {r_level}. "
                f"Phân rã thành 3 chỉ số nguy cơ: Tự động hóa tác vụ {sub_auto:.1f}%, "
                f"Độ chín muồi của công cụ AI thay thế {sub_tool:.1f}%, "
                f"Mức độ phơi nhiễm trước các tác tử tự chủ (Agentic Exposure) {sub_agentic:.1f}%. "
                f"Nhận định chuyên môn: {insight}"
            )
            corpus.append({
                "chunk_id": f"soc_{soc_code_str}_threat",
                "soc_code": soc_code_str,
                "major_group": major_group,
                "role_title": soc_title,
                "aspect": "threat_mechanism",
                "threat_score": t_score,
                "risk_level": r_level,
                "sub_scores": {
                    "task_automation": sub_auto,
                    "tool_maturity": sub_tool,
                    "agentic_exposure": sub_agentic,
                },
                "content": threat_content,
            })

            # Aspect 2: Defense Skills
            d_str = ", ".join([f"{i+1}. {s}" for i, s in enumerate(defense_skills)])
            defense_content = (
                f"Bộ 3 kỹ năng phòng thủ AI độc quyền của con người cho {soc_title} (SOC {soc_code_str}): {d_str}. "
                f"Đây là các năng lực tư duy phi tuyến tính, trí tuệ cảm xúc và phối hợp đa chiều mà các hệ thống "
                f"LLM và tác tử AI tự chủ chưa thể thay thế trong chu kỳ 1-3 năm tới."
            )
            corpus.append({
                "chunk_id": f"soc_{soc_code_str}_defense",
                "soc_code": soc_code_str,
                "major_group": major_group,
                "role_title": soc_title,
                "aspect": "defense_skills",
                "defense_skills": defense_skills,
                "content": defense_content,
            })

            # Aspect 3: Tasks and Tools
            tools_str = ", ".join(hot_tech[:6]) if hot_tech else "Công cụ nghiệp vụ chuyên ngành"
            tasks_str = "; ".join(tasks[:2]) if tasks else "Thực thi và giám sát quy trình nghiệp vụ"
            tasks_tools_content = (
                f"Tác vụ tự động hóa và Công cụ chuyển đổi của {soc_title} (SOC {soc_code_str}): "
                f"Các công cụ công nghệ bùng nổ nhu cầu (Hot Technologies) cần thành thạo để chuyển dịch gồm: {tools_str}. "
                f"Các tác vụ nghiệp vụ quan trọng hàng đầu cần tập trung nâng cao năng lực: {tasks_str}."
            )
            corpus.append({
                "chunk_id": f"soc_{soc_code_str}_tasks_tools",
                "soc_code": soc_code_str,
                "major_group": major_group,
                "role_title": soc_title,
                "aspect": "tasks_and_tools",
                "hot_technologies": hot_tech[:8],
                "critical_tasks": tasks[:3],
                "content": tasks_tools_content,
            })

        self.rag_corpus_ = corpus
        logger.info(
            "Hierarchical RAG Corpus generated successfully: %d micro-chunks tagged with major_group.",
            len(corpus),
        )
        return self.rag_corpus_
