"""
Fast LLM Profile Extractor Adapter.
Implements IProfileExtractor protocol:
- Uses DeepSeek Flash (Tier 1 Fast Extraction) via ChatOpenAI (base_url: https://api.deepseek.com).
- Supports automatic fallback to Google Gemini 1.5 Flash if DeepSeek key is absent.
- Enforces strict Pydantic v2 structured output with CandidateProfile.
- Raises domain CVExtractionError upon LLM inference or schema failure.
"""

from typing import Optional, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger
from source.domain.interfaces.parser import IProfileExtractor
from source.domain.entities.candidate import CandidateProfile
from source.domain.exceptions import CVExtractionError

logger = get_logger(__name__)

SYSTEM_EXTRACTION_PROMPT = """You are a Senior Technical Recruiter and Expert Resume Parsing Engine.
Your task is to analyze the provided candidate resume text and extract a comprehensive, structured candidate profile conforming strictly to the CandidateProfile schema.

Guidelines for extraction:
1. 'full_name': Extract candidate's full professional name found near the header. If missing, return null.
2. 'current_title': Most recent or current professional title (e.g., 'Backend Engineer', 'Data Analyst'). If fresh graduate, use 'Fresher / Graduate'.
3. 'years_of_experience': Accurately compute total cumulative professional experience in years from the employment timeline dates (e.g., 2021 - 2024 is ~3.0 years). For students or beginners, specify 0.0.
4. 'hard_skills': Core technical methodologies, domain principles, and conceptual capabilities (e.g., 'Microservices Architecture', 'Machine Learning', 'Data Modeling', 'RESTful API Design', 'Distributed Systems', 'Agile/Scrum').
5. 'soft_skills': Cognitive, interpersonal, and communication competencies (e.g., 'Problem Solving', 'Critical Thinking', 'Team Collaboration', 'Mentorship', 'Conflict Resolution').
6. 'tools_and_technologies': Concrete programming languages, frameworks, cloud platforms, database engines, and software tools (e.g., 'Python', 'FastAPI', 'Docker', 'PostgreSQL', 'PyTorch', 'AWS', 'Kubernetes', 'Redis', 'Git').
7. 'experiences': Chronological employment history. For each role:
   - 'company': Employer name
   - 'role_title': Job title
   - 'duration_months': Tenure in months (e.g., 1 year = 12 months)
   - 'description': Summary of responsibilities
   - 'key_achievements': List of notable milestones, performance metrics, and projects delivered
8. 'education': Academic degrees, universities, and graduation status (e.g., 'Bachelor of Science in Computer Science, HUST').
9. 'summary': Executive biography or stated career objective.
"""


class LLMProfileParser(IProfileExtractor):
    """
    Tier 1 Fast Extraction Client.
    Employs DeepSeek Flash (or fallback Gemini Flash) with temperature=0.0 for zero-hallucination structured parsing.
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Initialize LLMProfileParser with injected LLM or default configured provider.

        Args:
            llm: Optional LangChain chat model instance (useful for unit testing with MockLLM).
        """
        if llm is not None:
            self.llm = llm
            logger.info("LLMProfileParser initialized with custom injected LLM client.")
        else:
            self.llm = self._create_default_llm()

    def _create_default_llm(self) -> Any:
        """
        Factory helper to create the designated Tier 1 Fast Extraction LLM.
        Primary: DeepSeek Flash (via OpenAI compatible API)
        Fallback: Google Gemini 1.5 Flash
        """
        # 1. Primary: DeepSeek Flash / OpenRouter
        if settings.FAST_LLM_PROVIDER.lower() == "deepseek" and settings.DEEPSEEK_API_KEY:
            base_url = settings.DEEPSEEK_BASE_URL
            model = settings.FAST_LLM_MODEL
            # Auto-detect OpenRouter API key format
            if settings.DEEPSEEK_API_KEY.startswith("sk-or-v1-"):
                if "api.deepseek.com" in base_url:
                    base_url = "https://openrouter.ai/api/v1"
                if model == "deepseek-chat":
                    model = "deepseek/deepseek-chat"
            logger.info("Initializing DeepSeek client (model: %s, base_url: %s)...", model, base_url)
            return ChatOpenAI(
                model=model,
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=base_url,
                temperature=0.0,
                max_retries=3,
            )

        # 2. Fallback: Google Gemini Flash
        if settings.GEMINI_API_KEY:
            logger.info("Initializing Google Gemini Flash client as fallback (model: gemini-1.5-flash)...")
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.0,
                max_retries=3,
            )

        # 3. Fallback: Standard OpenAI if configured
        if settings.OPENAI_API_KEY:
            logger.info("Initializing OpenAI client as fallback (model: gpt-4o-mini)...")
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.OPENAI_API_KEY,
                temperature=0.0,
                max_retries=3,
            )

        # If no credentials found, instantiate DeepSeek placeholder for graceful offline handling
        logger.warning("No active LLM API keys detected in environment (DEEPSEEK_API_KEY or GEMINI_API_KEY).")
        return ChatOpenAI(
            model=settings.FAST_LLM_MODEL,
            api_key=settings.DEEPSEEK_API_KEY or "dummy_key",
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=0.0,
        )

    def extract_profile(self, raw_text: str) -> CandidateProfile:
        """
        Transforms raw resume text into standardized CandidateProfile entity.

        Args:
            raw_text: Extracted text from resume document.

        Returns:
            CandidateProfile: Validated domain entity.

        Raises:
            CVExtractionError: If input is empty or LLM extraction fails.
        """
        if not raw_text or len(raw_text.strip()) < 20:
            logger.error("Resume text is empty or too short (< 20 characters) for profile extraction.")
            raise CVExtractionError("Resume text is empty or too short to extract candidate profile.")

        logger.info("Starting LLM profile extraction for resume text (%d characters)...", len(raw_text))

        messages = [
            SystemMessage(content=SYSTEM_EXTRACTION_PROMPT),
            HumanMessage(content=f"Candidate Resume Raw Content:\n\n{raw_text}"),
        ]

        try:
            # Bind schema via with_structured_output
            structured_llm = self.llm.with_structured_output(CandidateProfile)
            result = structured_llm.invoke(messages)

            if isinstance(result, CandidateProfile):
                profile = result
            elif isinstance(result, dict):
                profile = CandidateProfile(**result)
            else:
                logger.error("LLM returned unexpected output type: %s", type(result))
                raise CVExtractionError(f"Unexpected output type from LLM: {type(result)}")

            logger.info(
                "Profile successfully extracted for '%s' (Title: '%s', Exp: %.1f years, Skills: %d items).",
                profile.full_name or "N/A",
                profile.current_title or "N/A",
                profile.years_of_experience,
                len(profile.get_all_skills),
            )
            return profile

        except CVExtractionError:
            raise
        except Exception as e:
            logger.exception("Error during LLM profile extraction: %s", str(e))
            raise CVExtractionError(f"Failed to extract structured profile via Fast LLM: {str(e)}") from e
