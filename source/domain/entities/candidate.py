from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class SkillItem(BaseModel):
    """Chi tiết về từng kỹ năng cụ thể của ứng viên."""
    name: Optional[str] = Field(..., description="Name of skill or technology (e.g., Python, SQL, Communication)")
    category: str = Field(..., description="Classification: hard_skill, soft_skill, tool (tool/software)")
    proficiency_level: Optional[str] = Field(default="", description="Proficiency level: beginner, intermediate, advanced")

    model_config = ConfigDict(extra="ignore")


class WorkExperience(BaseModel):
    """Thông tin về một vị trí công việc trong lịch sử làm việc."""
    company: Optional[str] = Field(default="N/A", description="Name of company or employing organization")
    role_title: str = Field(..., description="Job title (e.g., Backend Engineer)")
    duration_months: Optional[int] = Field(default=0, ge=0, description="Duration of employment (in months)")
    description: Optional[str] = Field(default="", description="Job description and key responsibilities")
    key_achievements: List[str] = Field(default_factory=list, description="List of notable achievements or completed projects")

    model_config = ConfigDict(extra="ignore")


class CandidateProfile(BaseModel):
    """
    Thực thể cốt lõi đại diện cho toàn bộ hồ sơ của ứng viên.
    Dữ liệu này được bóc tách từ CV và dùng xuyên suốt cho các tầng Router, ML, RAG và Copilot.
    """
    full_name: Optional[str] = Field(default=None, description="Candidate's full name")
    current_title: Optional[str] = Field(default=None, description="Current or most recent job title (leave blank if a student or inexperienced)")
    years_of_experience: float = Field(
        default=0.0,
        ge=0.0,
        alias="year_of_experience",
        description="Total years of accumulated work experience"
    )
    hard_skills: List[str] = Field(default_factory=list, description="List of hard skills (e.g., Machine Learning, Data Modeling)")
    soft_skills: List[str] = Field(default_factory=list, description="List of soft skills (e.g., Critical Thinking, Team Leadership)")
    tools_and_technologies: List[str] = Field(
        default_factory=list,
        description="List of specific tools, languages, and frameworks (e.g., Docker, PyTorch, AWS, FastAPI)"
    )
    experiences: List[WorkExperience] = Field(default_factory=list, description="List of work history")
    education: List[str] = Field(default_factory=list, description="Degree or educational qualification (e.g., Bachelor of Computer Science)")
    summary: Optional[str] = Field(default="", description="Personal summary or career objective")

    # Enriched Context Fields for RAG & LLM Countermeasures
    raw_text: Optional[str] = Field(
        default="",
        description="Preserved full raw resume text for factual RAG semantic retrieval and LLM context"
    )
    projects: List[str] = Field(
        default_factory=list,
        description="List of key engineering projects, systems built, or notable responsibilities"
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="List of professional certifications achieved (e.g., AWS Solutions Architect, CKA)"
    )
    domain_expertise: List[str] = Field(
        default_factory=list,
        description="Industry or functional domain expertise (e.g., FinTech, Cloud Infrastructure, E-commerce)"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore"
    )

    @property
    def year_of_experience(self) -> float:
        """Alias property for backward compatibility."""
        return self.years_of_experience

    @property
    def get_all_skills(self) -> List[str]:
        """Gộp toàn bộ kỹ năng và công cụ của ứng viên (loại bỏ trùng lặp)."""
        combined = set(self.hard_skills + self.soft_skills + self.tools_and_technologies)
        return list(combined)

    def to_rag_search_context(self) -> str:
        """
        Creates a condensed, high-signal semantic query context for O*NET vector retrieval.
        Combines current role, core mastered tools, top skills, and key projects.
        """
        parts = []
        if self.current_title:
            parts.append(f"Current Role: {self.current_title}")
        if self.tools_and_technologies:
            parts.append(f"Mastered Tools: {', '.join(self.tools_and_technologies[:15])}")
        if self.hard_skills:
            parts.append(f"Core Competencies: {', '.join(self.hard_skills[:10])}")
        if self.projects:
            parts.append(f"Projects: {'; '.join(self.projects[:2])}")
        return " | ".join(parts)
