"""
Deterministic (Non-LLM) CV Profile Extractor.
High-performance, zero-latency resume parser implementing IProfileExtractor:
- Extracts CandidateProfile deterministically using layout heuristics, regex pattern matching,
  and fuzzy O*NET market job title matching (< 5ms per resume).
- Fully decoupled from any external LLM APIs, eliminating network latency, rate limits, and token costs.
"""

import re
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from source.domain.entities.candidate import CandidateProfile, WorkExperience
from source.domain.interfaces.parser import IProfileExtractor
from source.domain.exceptions import CVExtractionError
from source.infrastructure.database.repo.onet_repository import OnetRepository
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class DeterministicProfileParser(IProfileExtractor):
    """
    Deterministic rule-based CV profile extractor.
    Operates in < 5ms without external LLM API calls.
    """

    TECH_SKILLS_KEYWORDS = {
        # Programming Languages
        "python", "java", "c++", "c#", "golang", "rust", "javascript", "typescript",
        "php", "ruby", "scala", "kotlin", "swift", "r", "dart",
        # Data & Databases
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle",
        "sqlite", "cassandra", "dynamodb", "snowflake", "bigquery",
        # Cloud & DevOps
        "docker", "kubernetes", "aws", "gcp", "azure", "linux", "git", "ci/cd",
        "terraform", "ansible", "jenkins", "gitlab", "github actions", "devops",
        # Web Frameworks
        "fastapi", "django", "flask", "spring boot", "react", "angular", "vue",
        "next.js", "node.js", "express", "html", "css", "tailwind", "graphql",
        "rest api", "microservices", "kafka", "spark", "hadoop", "airflow",
        # AI & Machine Learning
        "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "scipy",
        "xgboost", "llm", "langchain", "rag", "transformers", "opencv", "nlp",
        # Business Intelligence & Tools
        "tableau", "power bi", "excel", "jira", "confluence", "agile", "scrum",
    }

    SOFT_SKILLS_KEYWORDS = {
        "problem solving": "Problem Solving",
        "critical thinking": "Critical Thinking",
        "communication": "Effective Communication",
        "leadership": "Technical Leadership",
        "teamwork": "Cross-functional Teamwork",
        "collaboration": "Team Collaboration",
        "adaptability": "Adaptability",
        "time management": "Time Management",
        "mentoring": "Mentorship",
        "creative thinking": "Creative Thinking",
        "decision making": "Strategic Decision Making",
        "analytical": "Analytical Thinking",
    }

    TITLE_KEYWORDS = [
        "software engineer", "backend engineer", "frontend developer", "full stack developer",
        "data engineer", "data analyst", "data scientist", "machine learning engineer",
        "ai engineer", "devops engineer", "cloud architect", "solutions architect",
        "systems engineer", "qa engineer", "database administrator", "product manager",
        "scrum master", "technical lead", "developer", "architect", "analyst", "consultant",
        "accountant", "auditor", "financial analyst", "ui/ux designer", "ux designer",
        "ui designer", "product designer", "graphic designer", "web designer", "job title"
    ]

    def __init__(self, onet_repo: Optional[OnetRepository] = None) -> None:
        """
        Initializes the parser with an optional OnetRepository for title fuzzy matching.
        """
        self.onet_repo = onet_repo or OnetRepository()
        logger.debug("DeterministicProfileParser initialized successfully.")

    def extract_profile(self, raw_text: str) -> CandidateProfile:
        """
        Extracts structured CandidateProfile from raw resume text in < 5ms.
        """
        if not raw_text or len(raw_text.strip()) < 30:
            raise CVExtractionError("Resume text is too short or empty for structured extraction.")

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        lower_text = raw_text.lower()

        # 1. Contact Information
        email = self._extract_email(raw_text)
        phone = self._extract_phone(raw_text)

        # 2. Candidate Full Name
        full_name = self._extract_full_name(lines)

        # 3. Current Professional Title
        current_title = self._extract_job_title(lines, raw_text)

        # 4. Total Professional Experience (Years)
        years_of_experience = self._calculate_years_of_experience(lines, lower_text)

        # 5. Technical Skills, Tools & Soft Skills
        tools_and_tech = self._extract_tools(lower_text)
        hard_skills = tools_and_tech[:8]
        soft_skills = self._extract_soft_skills(lower_text)

        # 6. Education
        education = self._extract_education(lines)

        # 7. Projects, Experiences, Certifications, Domain
        projects = self._extract_projects(lines)
        experiences = self._extract_experiences(lines, current_title, years_of_experience)
        certifications = self._extract_certifications(lower_text)
        domain_expertise = self._extract_domain_expertise(lower_text)

        # Dynamic context-aware summary
        if current_title and years_of_experience > 0:
            summary = f"Chuyên viên {current_title} với {years_of_experience:.1f} năm kinh nghiệm chuyên môn."
        elif education:
            summary = f"Sinh viên / Fresher ngành {education[0]} với nền tảng kỹ năng: {', '.join(tools_and_tech[:4]) if tools_and_tech else 'kỹ thuật cơ bản'}."
        else:
            summary = f"Ứng viên với năng lực kỹ thuật: {', '.join(tools_and_tech[:4]) if tools_and_tech else 'Đang cập nhật'}."

        profile = CandidateProfile(
            full_name=full_name,
            current_title=current_title,
            years_of_experience=years_of_experience,
            hard_skills=hard_skills,
            soft_skills=soft_skills,
            tools_and_technologies=tools_and_tech,
            education=education,
            experiences=experiences,
            projects=projects,
            certifications=certifications,
            domain_expertise=domain_expertise,
            raw_text=raw_text,
            summary=summary,
        )

        logger.debug(
            "Deterministic extraction complete: '%s' | '%s' | %.1f yrs | %d tools | %d projects",
            profile.full_name,
            profile.current_title,
            profile.years_of_experience,
            len(profile.tools_and_technologies),
            len(profile.projects),
        )
        return profile

    def _extract_email(self, text: str) -> Optional[str]:
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        match = re.search(r"(\+?\d[\d\s\-\.\(\)]{8,18}\d)", text)
        return match.group(0) if match else None

    def _extract_full_name(self, lines: List[str]) -> str:
        """
        Extracts candidate name from the top header lines using negative filtering and pattern checks.
        """
        for line in lines[:10]:
            # Clean common prefixes/suffixes
            line_clean = re.sub(r"^(last updated.*?\||resume|curriculum vitae|cv)", "", line, flags=re.I).strip()
            if not line_clean:
                continue

            segments = [s.strip() for s in line_clean.split("|") if s.strip()]
            for seg in segments:
                # Remove extra spaces/dashes
                seg_clean = re.sub(r"[\-\–\—].*$", "", seg).strip()
                # Check for name pattern: 2 to 4 capitalized words
                if re.match(r"^[A-Z][a-zA-Z\.\-']+(\s+[A-Z][a-zA-Z\.\-']+){1,3}$", seg_clean):
                    lower_seg = seg_clean.lower()
                    # Exclude common tech/section words
                    if not any(k in lower_seg for k in [
                        "resume", "curriculum", "email", "phone", "profile", "summary",
                        "experience", "education", "skills", "updated", "san francisco",
                        "vietnam", "github", "linkedin", "engineer", "developer", "analyst",
                        "tableau", "microsoft", "google", "contact"
                    ]):
                        return seg_clean

        # Fallback to first line if reasonably short and capitalized
        if lines:
            first = lines[0].split("|")[0].strip()
            if len(first.split()) <= 4 and re.match(r"^[A-Z]", first):
                return first

        return "Ứng viên"

    def _extract_job_title(self, lines: List[str], raw_text: str) -> Optional[str]:
        """
        Extracts the candidate's current or target job title using keyword hierarchy
        and O*NET fuzzy matching against the header and experience sections. Returns None if unidentifiable.
        """
        lower_raw = raw_text.lower()

        # 1. Search for explicit title keywords in the top 15 lines
        for line in lines[:15]:
            if any(char in line for char in ["@", "http", "www."]):
                continue
            for part in re.split(r"[|•·\n]", line):
                clean_part = part.strip()
                if 4 <= len(clean_part) <= 50:
                    clean_lower = clean_part.lower()
                    for kw in self.TITLE_KEYWORDS:
                        if kw in clean_lower:
                            return clean_part.title()

        # 2. Search entire resume for high-signal title phrases
        for kw in self.TITLE_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower_raw):
                return kw.title()

        # 3. Search in experience sections for job titles
        in_exp = False
        for line in lines:
            line_l = line.lower().strip()
            if any(h in line_l for h in ["experience", "employment", "kinh nghiệm", "lịch sử làm việc"]):
                in_exp = True
                continue
            if in_exp:
                if any(h in line_l for h in ["education", "skills", "projects", "certifications", "học vấn", "kỹ năng"]):
                    break
                for kw in self.TITLE_KEYWORDS:
                    if kw in line_l and len(line) <= 60:
                        return line.title()

        # 4. Match against student / intern markers
        student_match = re.search(r"\b(intern|fresher|student|sinh viên|thực tập sinh|research intern)\b", lower_raw)
        if student_match:
            return student_match.group(0).title()

        # 5. Header snippet match against O*NET catalog
        header_snippet = " ".join(lines[:5])
        soc, matched_title, conf = self.onet_repo.match_job_title(header_snippet)
        if conf >= 0.70 and matched_title:
            return matched_title

        return None

    def _calculate_years_of_experience(self, lines: List[str], lower_text: str) -> float:
        """
        Estimates total cumulative years of experience from employment date ranges.
        Carefully excludes university study date ranges under the EDUCATION section.
        Defaults to 0.0 for students / freshers with no formal employment ranges.
        """
        current_year = datetime.datetime.now().year

        # Filter out lines belonging to EDUCATION section
        filtered_lines = []
        in_education = False
        for line in lines:
            line_lower = line.lower().strip()
            if any(h == line_lower or line_lower.startswith(h) for h in ["education", "học vấn", "academic"]):
                in_education = True
                continue
            if in_education:
                if any(h == line_lower or line_lower.startswith(h) for h in ["experience", "work experience", "employment", "kinh nghiệm", "projects", "dự án", "skills", "kỹ năng"]):
                    in_education = False
                else:
                    continue
            filtered_lines.append(line)

        target_text = "\n".join(filtered_lines).lower()

        # Find all 4-digit year ranges like "2018 - 2023" or "2020 - Present"
        year_matches = re.findall(
            r"\b(20\d\d|19\d\d)\b\s*[\-\–\—\to\s]+\s*\b(20\d\d|present|current|now)\b",
            target_text,
        )

        valid_ranges = []
        if year_matches:
            for start_y, end_y in year_matches:
                s_y = int(start_y)
                e_y = current_year if any(w in end_y for w in ["present", "current", "now"]) else int(end_y)
                if 1990 <= s_y <= current_year and s_y <= e_y <= current_year:
                    valid_ranges.append((s_y, e_y))

        if valid_ranges:
            min_year = min(r[0] for r in valid_ranges)
            max_year = max(r[1] for r in valid_ranges)
            if min_year <= max_year:
                return float(max(1.0, max_year - min_year))

        # Check for explicit phrases like "3+ years of experience"
        phrase_match = re.search(r"(\d+(\.\d+)?)\+?\s*years?(?:\s*of)?\s*experience", target_text)
        if phrase_match:
            return float(phrase_match.group(1))

        return 0.0


    def _extract_tools(self, lower_text: str) -> List[str]:
        """
        Scans document text against our comprehensive tech tool taxonomy.
        """
        found = set()
        for kw in self.TECH_SKILLS_KEYWORDS:
            pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
            if re.search(pattern, lower_text):
                found.add(kw.title() if len(kw) > 3 else kw.upper())

        # Sort alphabetically for consistency
        return sorted(list(found))

    def _extract_soft_skills(self, lower_text: str) -> List[str]:
        """
        Scans document text for key soft skills. Returns empty list if none explicitly mentioned.
        """
        found = []
        for kw, official_name in self.SOFT_SKILLS_KEYWORDS.items():
            if kw in lower_text:
                found.append(official_name)

        return found[:5]

    def _extract_education(self, lines: List[str]) -> List[str]:
        """
        Extracts academic degrees and university affiliations. Returns empty list if none found.
        """
        education_lines = []
        for line in lines:
            lower = line.lower()
            if any(deg in lower for deg in [
                "bachelor", "master", "b.s.", "m.s.", "ph.d", "b.a.",
                "university", "college", "institute", "degree", "polytechnic", "hust", "uet",
                "đại học", "học viện", "cao đẳng"
            ]):
                if 6 <= len(line) <= 120 and not any(k in lower for k in ["job", "project", "work"]):
                    education_lines.append(line.strip())

        return education_lines[:2]

    def _extract_projects(self, lines: List[str]) -> List[str]:
        """
        Extracts notable project titles and descriptions using section heuristics.
        """
        projects = []
        in_project_section = False

        for line in lines:
            line_clean = line.strip()
            lower = line_clean.lower()

            if any(h in lower for h in ["projects", "selected projects", "key projects", "notable projects", "dự án"]):
                in_project_section = True
                continue

            if in_project_section:
                # Detect boundary to next section
                if any(h in lower for h in ["education", "skills", "experience", "certifications", "contact", "học vấn"]):
                    in_project_section = False
                    break

                # Extract project title or bullet
                if line_clean.startswith(("-", "•", "*", "–")):
                    clean_bullet = line_clean.lstrip("-•*– ").strip()
                    if 15 <= len(clean_bullet) <= 180:
                        projects.append(clean_bullet)
                elif 5 <= len(line_clean) <= 60 and not line_clean.endswith("."):
                    projects.append(line_clean)

                if len(projects) >= 4:
                    break

        if not projects:
            # Fallback heuristic: check for action verbs describing technical achievements
            for line in lines:
                lower = line.lower()
                if any(act in lower for act in ["built", "designed", "developed", "architected", "deployed", "implemented", "optimized"]):
                    if 20 <= len(line.strip()) <= 150:
                        projects.append(line.strip().lstrip("-•*– "))
                        if len(projects) >= 3:
                            break

        return projects[:4]

    def _extract_experiences(self, lines: List[str], current_title: Optional[str], years: float) -> List[WorkExperience]:
        """
        Extracts structured work experiences from resume text. Returns empty list if no work history.
        """
        experiences = []
        achievements = []
        found_company: Optional[str] = None

        for line in lines:
            line_clean = line.strip()
            lower = line_clean.lower()

            # Look for company identifiers
            if any(c in lower for c in ["inc.", "corp", "corporation", "technologies", "solutions", "bank", "global", "group", "ltd", "công ty"]):
                if len(line_clean) < 50:
                    found_company = line_clean

            # Look for bullet points with achievements
            if line_clean.startswith(("-", "•", "*", "–")):
                bullet = line_clean.lstrip("-•*– ").strip()
                if 20 <= len(bullet) <= 200:
                    achievements.append(bullet)

        # Only construct WorkExperience if candidate actually has employment years or an identifiable company
        if (years > 0 or found_company) and current_title:
            duration_months = int(years * 12) if years > 0 else 6
            primary_exp = WorkExperience(
                company=found_company or "N/A",
                role_title=current_title,
                duration_months=duration_months,
                description=f"Kinh nghiệm làm việc chuyên môn tại {found_company or current_title}.",
                key_achievements=achievements[:4],
            )
            experiences.append(primary_exp)

        return experiences

    def _extract_certifications(self, lower_text: str) -> List[str]:
        """
        Extracts professional certifications recognized in the IT and O*NET domain.
        """
        cert_database = [
            ("aws certified solutions architect", "AWS Certified Solutions Architect"),
            ("aws certified developer", "AWS Certified Developer - Associate"),
            ("aws certified cloud practitioner", "AWS Certified Cloud Practitioner"),
            ("certified kubernetes administrator", "Certified Kubernetes Administrator (CKA)"),
            ("certified kubernetes application developer", "Certified Kubernetes Application Developer (CKAD)"),
            ("cka", "Certified Kubernetes Administrator (CKA)"),
            ("azure solutions architect", "Microsoft Certified: Azure Solutions Architect Expert"),
            ("azure administrator", "Microsoft Certified: Azure Administrator Associate"),
            ("google cloud professional", "Google Cloud Professional Cloud Architect"),
            ("gcp professional", "Google Cloud Professional Cloud Architect"),
            ("terraform associate", "HashiCorp Certified: Terraform Associate"),
            ("pmp", "Project Management Professional (PMP)"),
            ("cissp", "Certified Information Systems Security Professional (CISSP)"),
            ("scrum master", "Certified ScrumMaster (CSM)"),
        ]

        found = []
        for kw, official in cert_database:
            pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
            if re.search(pattern, lower_text):
                if official not in found:
                    found.append(official)

        return found[:4]

    def _extract_domain_expertise(self, lower_text: str) -> List[str]:
        """
        Identifies functional industry domains from candidate profile. Returns empty list if none detected.
        """
        domain_patterns = [
            (r"\b(fintech|banking|financial services|trading|payment)\b", "FinTech & Banking"),
            (r"\b(e-commerce|ecommerce|retail|marketplace)\b", "E-Commerce & Digital Retail"),
            (r"\b(healthcare|hospital|medical|biotech)\b", "Healthcare & Life Sciences"),
            (r"\b(cloud|saas|infrastructure|devops|platform)\b", "Cloud Infrastructure & Enterprise SaaS"),
            (r"\b(telecom|telecommunication|network|5g)\b", "Telecommunications"),
            (r"\b(automotive|iot|embedded|hardware)\b", "Automotive & IoT Systems"),
            (r"\b(security|cybersecurity|infosec)\b", "Cybersecurity & InfoSec"),
            (r"\b(data platform|big data|analytics)\b", "Big Data & Analytical Systems"),
        ]

        domains = []
        for pat, label in domain_patterns:
            if re.search(pat, lower_text):
                domains.append(label)

        return domains[:3]


