"""
Database / Storage Record Models for O*NET Data.
Defines schemas for metadata catalog and feature store records.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class OnetMetadataRecord(BaseModel):
    """Schema representing an occupation metadata record in persistent storage."""
    soc_code: str
    soc_title: str
    major_group: str
    major_group_name: str
    threat_score: float
    risk_level: str
    sub_task_automation: float = 0.0
    sub_tool_maturity: float = 0.0
    sub_agentic_exposure: float = 0.0
    defense_skills: List[str] = Field(default_factory=list)
    insight: Optional[str] = None
    median_salary: float = 0.0
    software_skills_catalog: List[str] = Field(default_factory=list)
    top_hot_technologies: List[str] = Field(default_factory=list)
    critical_tasks: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")
