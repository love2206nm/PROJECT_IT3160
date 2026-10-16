
from enum import Enum
# ép các biến chỉ được nhận 1 giá tri thôi
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class RiskLevel(str, Enum):
    """Categorical AI threat level based on O*NET benchmark percentiles."""
    LOW = "Low"               # Low exposure (Score 0 - 30)
    MODERATE = "Moderate"     # Moderate exposure (Score 31 - 50)
    HIGH = "High"             # High exposure (Score 51 - 70)
    VERY_HIGH = "Very High"   # Severe agentic threat (Score 71 - 100)


class SubRiskScores(BaseModel):
    """diem cho cac khia canh bi AI thay the"""
    task_automation: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Degree of routine daily task automation exposure (sub_task_automation)"
    )
    tool_maturity: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Maturity and capability level of existing industry AI tools (sub_tool_maturity)"
    )
    agentic_exposure: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Vulnerability and direct exposure to autonomous AI workflow agents (sub_agentic_exposure)"
    )

    model_config = ConfigDict(extra="ignore")


class RiskProfile(BaseModel):
    """
    Comprehensive risk profile for an occupational role.
    Returned either via Deterministic Lookup (exact SOC match) or ML XGBoost Regressor.
    """
    soc_code: Optional[str] = Field(
        default=None,
        description="Standard O*NET SOC occupational classification code (e.g., '15-1252.00')"
    )
    role_title: str = Field(
        ...,
        description="Occupational job title being evaluated"
    )
    threat_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall AI career threat index score from 0.0 to 100.0 (dataset range: 4 to 92)"
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical threat classification: Low, Moderate, High, or Very High"
    )
    sub_scores: SubRiskScores = Field(
        default_factory=SubRiskScores,
        description="Granular decomposition across task, tool, and agentic dimensions"
    )
    defense_skills: List[str] = Field(
        default_factory=list,
        description="Top 3 actionable AI defense competencies extracted from O*NET dataset"
    )
    insight: Optional[str] = Field(
        default=None,
        description="Expert qualitative assessment explaining the specific AI displacement mechanism"
    )
    is_deterministic: bool = Field(
        default=True,
        description="True if matched directly from authoritative database lookup (0ms), False if inferred via ML"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the title semantic match or ML risk prediction"
    )
    salary_trend: Optional[str] = Field(
        default=None,
        description="Observed market compensation trajectory: Growing, Stable, or Declining"
    )

    model_config = ConfigDict(
        use_enum_values=True,
        extra="ignore"
    )
