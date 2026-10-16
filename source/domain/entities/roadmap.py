from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class RoadmapPhase(BaseModel):
    """
    Một giai đoạn trong lộ trình hành động chuyển đổi nghề nghiệp (30, 60, hoặc 90 ngày).
    """
    phase_name: str = Field(
        ...,
        description="Name of transition phase (e.g., 'Phase 1: 30 Days - Defensive Foundation', 'Phase 2: 60 Days - Deepening', 'Phase 3: 90 Days - Mastery')"
    )
    duration_days: int = Field(
        default=30,
        ge=1,
        description="Duration allocated for this phase in days"
    )
    objective: str = Field(
        ...,
        description="Primary strategic milestone and goal to achieve within this timeframe"
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Actionable, concrete weekly tasks and milestones"
    )
    recommended_certifications: List[str] = Field(
        default_factory=list,
        description="Authoritative, industry-recognized certificates relevant to the target role"
    )
    target_skills: List[str] = Field(
        default_factory=list,
        description="Specific AI defense skills focused during this phase"
    )
    practical_projects: List[str] = Field(
        default_factory=list,
        description="Hands-on portfolio projects demonstrating applied mastery"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore"
    )


class CopilotRoadmap(BaseModel):
    """
    Báo cáo cố vấn nghề nghiệp toàn diện và lộ trình hành động cá nhân hóa do LLM Copilot Agent sinh ra.
    """
    target_role: str = Field(
        ...,
        description="Safe transition target career role"
    )
    target_soc_code: Optional[str] = Field(
        default=None,
        description="O*NET SOC occupational classification code for the target role"
    )
    executive_summary: str = Field(
        ...,
        description="High-level strategic narrative explaining the career defense rationale"
    )
    phases: List[RoadmapPhase] = Field(
        default_factory=list,
        description="Structured transition milestones (30 Days, 60 Days, 90 Days)"
    )
    defense_skills_breakdown: List[str] = Field(
        default_factory=list,
        description="Key AI defense competencies cultivated throughout this roadmap"
    )
    mermaid_chart: str = Field(
        default="",
        description="Valid Mermaid flowchart/gantt syntax for visual diagram rendering on frontend"
    )
    estimated_completion_months: int = Field(
        default=3,
        ge=1,
        description="Estimated timeline to achieve career transition readiness in months"
    )
    disclaimer: str = Field(
        default="This roadmap is generated with evidence-grounded AI intelligence based on O*NET benchmark standards. Real outcomes depend on candidate execution and labor market dynamics.",
        description="Advisory disclaimer and guidance notice"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore"
    )
