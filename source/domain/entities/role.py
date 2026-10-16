"""
O*NET Occupational Role Domain Entity.
Represents a standardized occupational profile from the 867 Base SOC metadata catalog.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from source.domain.entities.risk import RiskLevel


class OnetRole(BaseModel):
    """
    Core Domain Entity representing a standardized O*NET occupation.
    Encapsulates official classification, compensation, automation exposure, and defense competencies.
    """
    soc_code: str = Field(
        ...,
        description="Standard 6-digit O*NET SOC classification code, e.g., '15-1252'"
    )
    soc_title: str = Field(
        ...,
        description="Official occupational title, e.g., 'Software Developers'"
    )
    major_group: str = Field(
        ...,
        description="2-digit major occupation prefix, e.g., '15'"
    )
    major_group_name: str = Field(
        ...,
        description="Official major group description, e.g., 'Computer and Mathematical Occupations'"
    )
    median_salary: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Median annual compensation in USD after empirical metro pooling or fallback"
    )
    threat_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="AI Automation Threat Index Score [0.0 - 100.0]"
    )
    risk_level: Optional[RiskLevel] = Field(
        default=None,
        description="Categorical risk exposure classification: Low, Moderate, High, Very High"
    )
    sub_task_automation: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Decomposition score: routine task automation exposure"
    )
    sub_tool_maturity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Decomposition score: AI replacement tool maturity"
    )
    sub_agentic_exposure: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Decomposition score: autonomous AI agent exposure"
    )
    defense_skills: List[str] = Field(
        default_factory=list,
        description="Top 3 human defense skills insulating role from automation"
    )
    top_hot_technologies: List[str] = Field(
        default_factory=list,
        description="High-demand tools and software technologies required by role"
    )
    critical_tasks: List[str] = Field(
        default_factory=list,
        description="Top essential high-importance daily tasks"
    )
    software_skills_catalog: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Granular inventory of software tools classified by taxonomy category"
    )
    insight: Optional[str] = Field(
        default=None,
        description="Qualitative expert assessment explaining the AI displacement mechanism"
    )

    model_config = ConfigDict(
        use_enum_values=True,
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore"
    )
