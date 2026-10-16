from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

from source.domain.entities.risk import RiskLevel


class SafeTransitionRole(BaseModel):
    """
    Thực thể biểu diễn một vị trí nghề nghiệp an toàn lân cận (Safe Career Pivot).
    Được tính toán từ mô hình phân cụm Agglomerative Clustering và hàm thỏa dụng Utility.
    """
    target_soc_code: str = Field(
        ...,
        description="Standard O*NET SOC code for the target safe role (e.g., '15-1251.00')"
    )
    target_title: str = Field(
        ...,
        description="Official title of the target pivot occupation"
    )
    utility_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite transition utility score balancing skill overlap, AI safety, and salary advantage"
    )
    similarity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Cosine similarity percentage in the 120D O*NET skill-space between current and target role"
    )
    threat_reduction: float = Field(
        ...,
        description="Reduction in threat index points compared to current role (Current Threat - Target Threat)"
    )
    salary_advantage: float = Field(
        default=0.0,
        description="Relative compensation differential ratio or net advantage compared to baseline"
    )
    target_threat_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Baseline AI threat score of target role"
    )
    target_risk_level: Optional[RiskLevel] = Field(
        default=None,
        description="Categorical AI risk level of target role"
    )
    median_salary: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Real-adjusted national median salary for target role (USD/year)"
    )
    defense_skills: List[str] = Field(
        default_factory=list,
        description="Top 3 actionable AI defense competencies for this target role"
    )
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills already possessed by candidate that match this role"
    )
    missing_critical_skills: List[str] = Field(
        default_factory=list,
        description="Key competencies candidate must acquire to successfully transition"
    )
    matched_tools: List[str] = Field(
        default_factory=list,
        description="Software tools and technologies candidate already mastered matching target role (A ∩ B)"
    )
    missing_critical_tools: List[str] = Field(
        default_factory=list,
        description="Essential software tools required by target role that candidate lacks (B \\ A)"
    )
    hot_technologies_to_learn: List[str] = Field(
        default_factory=list,
        description="High-demand tools from O*NET Hot Technology list required by target role"
    )
    countermeasures: List[str] = Field(
        default_factory=list,
        description="Actionable strategic countermeasures mapping candidate's current strengths to bridge critical skill gaps"
    )

    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore"
    )


class SkillGap(BaseModel):
    """
    Phân tích khoảng cách kỹ năng (Skill Gap Analysis) chi tiết giữa hồ sơ ứng viên
    và vai trò an toàn mục tiêu.
    """
    target_role_title: str = Field(
        default="",
        description="Title of target occupational role being benchmarked"
    )
    target_soc_code: str = Field(
        default="",
        description="O*NET SOC code of target occupational role"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Existing transferable skills and competencies already mastered by candidate"
    )
    vulnerable_skills: List[str] = Field(
        default_factory=list,
        description="Candidate's current skills that face high AI automation obsolescence"
    )
    defense_skills: List[str] = Field(
        default_factory=list,
        description="Target high-leverage defense competencies insulating candidate from AI replacement"
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Technical tools, methodologies, or domain skills needed for target role"
    )
    matched_tools: List[str] = Field(
        default_factory=list,
        description="Software tools and technologies candidate already mastered matching target role (A ∩ B)"
    )
    missing_critical_tools: List[str] = Field(
        default_factory=list,
        description="Essential software tools required by target role that candidate lacks (B \\ A)"
    )
    hot_technologies_to_learn: List[str] = Field(
        default_factory=list,
        description="High-demand tools from O*NET Hot Technology list required by target role"
    )
    countermeasures: List[str] = Field(
        default_factory=list,
        description="Actionable strategic countermeasures mapping candidate's current strengths to bridge critical skill gaps"
    )
    overlap_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Total percentage of competency alignment between candidate and target role"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore"
    )

