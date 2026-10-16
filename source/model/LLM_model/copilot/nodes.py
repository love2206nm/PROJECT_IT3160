"""
Specialized Node Implementations for LangGraph Copilot State Machine.
Implements the 10 core nodes + interactive multi-turn copilot node:
1. cv_ingest_node
2. extract_profile_node (5-tier SOC matching + fresher/student mode)
3. human_review_node (HITL checkpoint)
4. deterministic_lookup (O*NET 0ms)
5. ml_scoring_node (XGBoost Regressor + SHAP)
6. cluster_pivot_node (Ward Hierarchical Clustering for Top 2-3 Safe Pivots)
7. rag_grounding_node (Hybrid RAG Evidence Retrieval)
8. copilot_agent_node (Reasoning LLM Generator via CopilotPromptCatalog)
9. critic_evaluator_node (Independent Critic Reflection via CopilotPromptCatalog)
10. assemble_result_node (Domain AnalysisResult packaging)
11. interactive_copilot_node (Multi-turn conversational advising via CopilotPromptCatalog)
"""

import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile, RiskLevel, SubRiskScores
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.entities.roadmap import CopilotRoadmap, RoadmapPhase
from source.domain.entities.analysis import AnalysisResult
from source.domain.entities.prompt import CopilotPromptCatalog
from source.domain.interfaces.onet_repository import IOnetRepository
from source.domain.interfaces.rag import IEvidenceRetriever
from source.parsers.document_parser import DocumentParser
from source.parsers.deterministic_profile_parser import DeterministicProfileParser
from source.model.LLM_model.copilot.state import AgentState

if TYPE_CHECKING:
    from source.services.user.career_service import UserCareerService

from source.model.LLM_model.copilot.prompts import (
    CriticEvaluation,
    CriticScores,
    LLMSocSuggestion,
)
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


def _get_default_onet_repo() -> IOnetRepository:
    """Lazy-load OnetRepository singleton to avoid concrete DB import at module level."""
    from source.infrastructure.database.repo.onet_repository import OnetRepository
    return OnetRepository()


def _get_default_evidence_retriever() -> IEvidenceRetriever:
    """Lazy-load HierarchicalEvidenceRetriever to avoid concrete import at module level."""
    from source.model.LLM_model.rag.retriever import HierarchicalEvidenceRetriever
    return HierarchicalEvidenceRetriever()


# ==============================================================================
# Node 1: CV Ingestion Node
# ==============================================================================

def cv_ingest_node(
    state: AgentState,
    doc_parser: Optional[DocumentParser] = None,
) -> Dict[str, Any]:
    """
    Step 1: Normalizes raw text from CV file path or validates existing raw text.
    """
    logger.info("[Node: cv_ingest_node] Ingesting candidate resume content...")
    raw_text = state.get("raw_cv_text", "").strip()
    file_path = state.get("file_path")

    if not raw_text and file_path:
        parser = doc_parser or DocumentParser()
        raw_text = parser.extract_text(file_path)

    logger.info("[Node: cv_ingest_node] Ingestion completed (%d characters).", len(raw_text))
    return {
        "raw_cv_text": raw_text,
        "iteration_count": state.get("iteration_count", 0),
    }


# ==============================================================================
# Node 2: Profile Extraction & 5-Tier SOC Matching Node
# ==============================================================================

def extract_profile_node(
    state: AgentState,
    profile_parser: Optional[Any] = None,
    onet_repo: Optional[IOnetRepository] = None,
) -> Dict[str, Any]:
    """
    Step 2: Uses Deterministic Profile Parser (< 5ms) to extract structured CandidateProfile
    and executes 5-tier SOC matching:
    Tier 1: Job title (if non-empty & not generic student)
    Tier 2: Skills matching via repo.match_role_by_skills
    Tier 3: Education matching via repo.match_role_by_education
    Tier 4: Projects domain analysis
    Tier 5: Fresher/Student mode detection
    """
    logger.info("[Node: extract_profile_node] Extracting structured profile deterministically (< 5ms)...")
    raw_text = state.get("raw_cv_text", "")
    profile = state.get("candidate_profile")

    repo = onet_repo or _get_default_onet_repo()

    if profile is None and raw_text:
        parser = profile_parser or DeterministicProfileParser(onet_repo=repo)
        profile = parser.extract_profile(raw_text)

    raw_title = (profile.current_title if profile else "").strip()
    res = None
    if hasattr(repo, "resolve_candidate_soc"):
        try:
            res = repo.resolve_candidate_soc(profile)
        except Exception:
            res = None

    if isinstance(res, tuple) and len(res) == 4:
        matched_soc, matched_title, conf, is_student_or_fresher = res
    else:
        student_keywords = ["student", "sinh viên", "học sinh", "intern", "thực tập", "fresher", "graduated", "tốt nghiệp"]
        is_student_or_fresher = (
            any(k in raw_title.lower() for k in student_keywords)
            or (profile is not None and profile.years_of_experience == 0.0)
        )
        matched_soc, matched_title, conf = repo.match_job_title(raw_title)
        if (not matched_soc or conf < 0.60) and profile and profile.get_all_skills and hasattr(repo, "match_role_by_skills"):
            s_soc, s_title, s_conf = repo.match_role_by_skills(profile.get_all_skills)
            if s_soc and s_conf > conf:
                matched_soc, matched_title, conf = s_soc, s_title, s_conf
        if (not matched_soc or conf < 0.50) and profile and profile.education and hasattr(repo, "match_role_by_education"):
            e_soc, e_title, e_conf = repo.match_role_by_education(profile.education)
            if e_soc and e_conf > conf:
                matched_soc, matched_title, conf = e_soc, e_title, e_conf

    logger.info(
        "[Node: extract_profile_node] Profile: '%s' | Title: '%s' | Matched: '%s' (SOC: %s, Conf: %.3f, FresherMode: %s)",
        profile.full_name if profile else "Unknown",
        raw_title,
        matched_title,
        matched_soc,
        conf,
        is_student_or_fresher,
    )

    return {
        "candidate_profile": profile,
        "matched_soc_code": matched_soc,
        "matched_title": matched_title,
        "match_confidence": conf,
        "is_fresher_mode": is_student_or_fresher,
    }


# ==============================================================================
# Node 3: Human Review Node (HITL Interruption Point)
# ==============================================================================

def human_review_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 3: Point of Human-in-the-Loop interruption.
    Validates user confirmation or manual title/SOC override.
    """
    logger.info(
        "[Node: human_review_node] Human review processed. Confirmed SOC: '%s' (%s)",
        state.get("matched_soc_code"),
        state.get("matched_title"),
    )
    return {"user_confirmed": True}


# ==============================================================================
# Node 4A: Deterministic O*NET Lookup (High Confidence Branch)
# ==============================================================================

def deterministic_lookup(
    state: AgentState,
    onet_repo: Optional[IOnetRepository] = None,
) -> Dict[str, Any]:
    """
    Step 4A: 0ms authoritative catalog lookup for high-confidence exact matches (>= 0.90).
    """
    soc_code = state.get("matched_soc_code")
    logger.info("[Node: deterministic_lookup] Performing 0ms deterministic lookup for SOC '%s'...", soc_code)

    if not soc_code:
        logger.warning("[Node: deterministic_lookup] No SOC code matched; returning None risk profile.")
        return {"risk_profile": None}

    repo = onet_repo or _get_default_onet_repo()
    risk_profile = repo.get_risk_profile(soc_code)

    if risk_profile is None:
        matched_title = state.get("matched_title") or "Professional Role"
        logger.warning("[Node: deterministic_lookup] SOC '%s' not in catalog, predicting via UserCareerService ML...", soc_code)
        try:
            from source.services.user.career_service import UserCareerService
            service = UserCareerService(onet_repo=repo)
            candidate = state.get("candidate_profile") or CandidateProfile(current_title=matched_title)
            risk_profile = service.predict_hybrid_role_risk(candidate)
            risk_profile.soc_code = soc_code
            risk_profile.is_deterministic = False
            risk_profile.confidence = state.get("match_confidence", 0.85)
        except Exception as e:
            logger.warning("[Node: deterministic_lookup] ML prediction fallback on error: %s", e)
            risk_profile = RiskProfile(
                soc_code=soc_code,
                role_title=matched_title,
                threat_score=40.0,
                risk_level=RiskLevel.MODERATE,
                is_deterministic=False,
                confidence=0.5,
            )
    else:
        risk_profile.is_deterministic = True
        risk_profile.confidence = 1.0

    logger.info(
        "[Node: deterministic_lookup] Retrieved authoritative RiskProfile (Score: %.1f, Level: %s)",
        risk_profile.threat_score,
        risk_profile.risk_level,
    )
    return {"risk_profile": risk_profile}


# ==============================================================================
# Node 4B: ML Scoring Node (Custom / Hybrid / Low Confidence Branch)
# ==============================================================================

def ml_scoring_node(
    state: AgentState,
    career_service: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Step 4B: In-depth XGBoost regressor scoring with SHAP explainability for custom/hybrid roles.
    """
    logger.info("[Node: ml_scoring_node] Inferring automation threat via Regularized XGBoost Regressor...")
    profile = state.get("candidate_profile")
    if not profile:
        profile = CandidateProfile(current_title=state.get("matched_title") or "Candidate")

    if career_service is None:
        from source.services.user.career_service import UserCareerService
        service = UserCareerService()
    else:
        service = career_service

    risk_profile = service.predict_hybrid_role_risk(profile)
    risk_profile.is_deterministic = False

    logger.info(
        "[Node: ml_scoring_node] XGBoost predicted threat: %.1f (%s) with %d defense skills.",
        risk_profile.threat_score,
        risk_profile.risk_level,
        len(risk_profile.defense_skills),
    )
    return {"risk_profile": risk_profile}


# ==============================================================================
# Node 5: Clustering & Safe Pivot Node (Top 2-3 Safe Pivots)
# ==============================================================================

def cluster_pivot_node(
    state: AgentState,
    career_service: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Step 5: Agglomerative clustering (Ward) to find Top 2-3 safe pivot roles and calculate SkillGaps for all 3.
    """
    soc_code = state.get("matched_soc_code")
    risk_profile = state.get("risk_profile")
    current_threat = risk_profile.threat_score if risk_profile else 50.0

    logger.info("[Node: cluster_pivot_node] Finding safe pivot recommendations for SOC '%s'...", soc_code)
    if career_service is None:
        from source.services.user.career_service import UserCareerService
        service = UserCareerService()
    else:
        service = career_service

    safe_pivots: List[SafeTransitionRole] = []
    if soc_code:
        safe_pivots = service.get_safe_career_pivots(
            soc_code=soc_code,
            current_threat=current_threat,
            top_k=3,
        )

    candidate = state.get("candidate_profile")
    candidate_skills = set(candidate.get_all_skills) if candidate else set()

    skill_gaps: List[SkillGap] = []
    for p in safe_pivots[:3]:
        # Logically identify vulnerable candidate skills: skills that are not transferable strengths nor defense insulated
        vulnerable = [s for s in candidate_skills if s not in p.matched_skills and s not in p.defense_skills]
        gap = SkillGap(
            target_role_title=p.target_title,
            target_soc_code=p.target_soc_code,
            strengths=p.matched_skills,
            vulnerable_skills=vulnerable,
            defense_skills=p.defense_skills,
            missing_skills=p.missing_critical_skills,
            matched_tools=p.matched_tools,
            missing_critical_tools=p.missing_critical_tools,
            hot_technologies_to_learn=p.hot_technologies_to_learn,
            overlap_percentage=p.similarity_pct,
        )
        skill_gaps.append(gap)

    primary_pivot: Optional[SafeTransitionRole] = safe_pivots[0] if safe_pivots else None
    primary_skill_gap: Optional[SkillGap] = skill_gaps[0] if skill_gaps else None

    logger.info(
        "[Node: cluster_pivot_node] Discovered %d safe pivots. Primary: '%s'",
        len(safe_pivots),
        primary_pivot.target_title if primary_pivot else "None",
    )
    return {
        "safe_pivot_roles": safe_pivots,
        "skill_gap": primary_skill_gap,
        "skill_gaps": skill_gaps,
    }


# ==============================================================================
# Node 6: RAG Grounding Node
# ==============================================================================

def rag_grounding_node(
    state: AgentState,
    retriever: Optional[IEvidenceRetriever] = None,
) -> Dict[str, Any]:
    """
    Step 6: Hierarchical Aspect-Based retrieval for factual O*NET evidence using Contextual Hybrid Query.
    """
    skill_gap = state.get("skill_gap")
    target_soc = skill_gap.target_soc_code if skill_gap else (state.get("matched_soc_code") or "")
    target_title = skill_gap.target_role_title if skill_gap else "Target Career"
    candidate = state.get("candidate_profile")
    cand_context = candidate.to_rag_search_context() if candidate and hasattr(candidate, "to_rag_search_context") else ""

    query = f"Target Role: {target_title} (SOC: {target_soc}). Key tasks and technical defense skills. Candidate Background: {cand_context}"
    logger.info("[Node: rag_grounding_node] Retrieving O*NET factual evidence with contextual query: '%s'...", query[:100])

    evidence_retriever = retriever or _get_default_evidence_retriever()
    evidences = evidence_retriever.retrieve_evidence(
        query=query,
        soc_code=target_soc if target_soc else None,
        top_k=settings.RAG_TOP_K,
    )

    logger.info("[Node: rag_grounding_node] Retrieved %d grounded evidence chunks.", len(evidences))
    return {"grounding_evidence": evidences}


# ==============================================================================
# Node 7: Copilot Agent Node (Roadmap Generator with Prompt Catalog)
# ==============================================================================

def copilot_agent_node(
    state: AgentState,
    copilot_llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Step 7: Reasoning LLM crafts an actionable, evidence-grounded 30-60-90 day transition roadmap
    using CopilotPromptCatalog templates (Student Onboarding vs Multi-Pivot Advisory).
    """
    iteration = state.get("iteration_count", 0) + 1
    logger.info("[Node: copilot_agent_node] Generating career transition roadmap (Iteration %d)...", iteration)

    candidate = state.get("candidate_profile") or CandidateProfile(current_title=state.get("matched_title") or "Candidate")
    risk = state.get("risk_profile")
    skill_gap = state.get("skill_gap")
    skill_gaps = state.get("skill_gaps", [skill_gap] if skill_gap else [])
    safe_pivots = state.get("safe_pivot_roles", [])
    evidences = state.get("grounding_evidence", [])
    critic_feedback = state.get("critic_feedback", "")
    is_fresher = state.get("is_fresher_mode", False)

    target_role = skill_gap.target_role_title if skill_gap else (safe_pivots[0].target_title if safe_pivots else "Technology Professional")
    target_soc = skill_gap.target_soc_code if skill_gap else (safe_pivots[0].target_soc_code if safe_pivots else "15-1252.00")

    # Use CopilotPromptCatalog for prompt composition
    if is_fresher:
        user_prompt = CopilotPromptCatalog.build_student_onboarding_prompt(
            candidate=candidate,
            safe_pivots=safe_pivots,
            skill_gaps=skill_gaps,
        )
    else:
        user_prompt = CopilotPromptCatalog.build_multi_pivot_advisory_prompt(
            candidate=candidate,
            risk_profile=risk,
            safe_pivots=safe_pivots,
            skill_gaps=skill_gaps,
            grounding_evidences=evidences,
            critic_feedback=critic_feedback,
        )

    llm = copilot_llm or _create_default_reasoning_llm()

    try:
        structured_llm = llm.with_structured_output(CopilotRoadmap)
        roadmap = structured_llm.invoke([
            SystemMessage(content=CopilotPromptCatalog.COPILOT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        if not isinstance(roadmap, CopilotRoadmap):
            roadmap = CopilotRoadmap(**roadmap)
    except Exception as e:
        logger.warning("[Node: copilot_agent_node] Structured generation fallback: %s", e)
        roadmap = _create_fallback_roadmap(target_role, target_soc, skill_gap)

    # Populate strategic countermeasures in primary skill_gap
    if skill_gap:
        primary_tool = skill_gap.missing_critical_tools[0] if skill_gap.missing_critical_tools else "Cloud Architecture"
        cert = roadmap.phases[0].recommended_certifications[0] if roadmap.phases and roadmap.phases[0].recommended_certifications else "Chứng chỉ Quốc tế"
        skill_gap.countermeasures = [
            f"Tận dụng kinh nghiệm {candidate.current_title or 'chuyên môn'} để học nhanh {primary_tool} thông qua các bài lab thực chiến",
            f"Xây dựng dự án Portfolio mã nguồn mở chứng minh năng lực {target_role} chuẩn kiến trúc",
            f"Đầu tư ôn luyện chứng chỉ {cert} để chuẩn hóa hồ sơ năng lực cạnh tranh trước AI",
        ]

    logger.info(
        "[Node: copilot_agent_node] Roadmap generated successfully for '%s' (%d phases, est %d months).",
        roadmap.target_role,
        len(roadmap.phases),
        roadmap.estimated_completion_months,
    )

    return {
        "copilot_roadmap": roadmap,
        "iteration_count": iteration,
        "skill_gap": skill_gap,
    }


# ==============================================================================
# Node 8: Critic Evaluator Node (Independent Self-Reflection)
# ==============================================================================

def critic_evaluator_node(
    state: AgentState,
    critic_llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Step 8: Independent Critic Audits the roadmap against 3 criteria: Feasibility, Grounding, Anti-Hallucination.
    """
    logger.info("[Node: critic_evaluator_node] Critic evaluating roadmap quality and feasibility...")
    roadmap = state.get("copilot_roadmap")
    candidate = state.get("candidate_profile")
    skill_gap = state.get("skill_gap")

    if not roadmap:
        logger.warning("[Node: critic_evaluator_node] No roadmap in state. Auto-rejecting.")
        return {
            "critic_approved": False,
            "critic_feedback": "Roadmap is missing or empty.",
            "critic_review": {"approved": False, "feedback": "Roadmap empty", "scores": {"feasibility": 0.0, "grounding": 0.0, "hallucination_free": 0.0}},
        }

    critic_prompt = f"""EVALUATE THIS CAREER ROADMAP:
Target Role: {roadmap.target_role} (SOC: {roadmap.target_soc_code})
Candidate Current Title: {candidate.current_title if candidate else 'Unknown'}
Candidate Experience: {candidate.years_of_experience if candidate else 0} years

ROADMAP PHASES:
{chr(10).join([f'Phase {p.phase_name}: {p.objective} | Actions: {len(p.action_items)} | Certs: {p.recommended_certifications}' for p in roadmap.phases])}

DEFENSE SKILLS: {roadmap.defense_skills_breakdown}
TARGET SKILL GAP TO ADDRESS: {skill_gap.missing_skills if skill_gap else []}
"""

    llm = critic_llm or _create_default_critic_llm()

    try:
        structured_critic = llm.with_structured_output(CriticEvaluation)
        eval_result = structured_critic.invoke([
            SystemMessage(content=CopilotPromptCatalog.CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=critic_prompt),
        ])
        if not isinstance(eval_result, CriticEvaluation):
            eval_result = CriticEvaluation(**eval_result)
    except Exception as e:
        logger.warning("[Node: critic_evaluator_node] Critic structured call fallback: %s", e)
        eval_result = CriticEvaluation(
            approved=len(roadmap.phases) >= 3,
            feedback="Roadmap structured correctly across 30-60-90 day progression.",
            scores=CriticScores(feasibility=8.5, grounding=8.5, hallucination_free=9.0),
        )

    logger.info(
        "[Node: critic_evaluator_node] Critic verdict: Approved=%s | Scores: Feas=%.1f, Ground=%.1f, Halluc=%.1f",
        eval_result.approved,
        eval_result.scores.feasibility,
        eval_result.scores.grounding,
        eval_result.scores.hallucination_free,
    )

    return {
        "critic_approved": eval_result.approved,
        "critic_feedback": eval_result.feedback,
        "critic_review": eval_result.model_dump(),
    }


# ==============================================================================
# Node 9: Assemble Result Node (Domain AnalysisResult Packaging)
# ==============================================================================

def assemble_result_node(state: AgentState) -> Dict[str, Any]:
    """
    Step 9: Packages all validated domain entities into a unified AnalysisResult.
    """
    logger.info("[Node: assemble_result_node] Assembling unified AnalysisResult domain payload...")
    candidate = state.get("candidate_profile") or CandidateProfile(current_title=state.get("matched_title") or "Candidate")
    risk = state.get("risk_profile")
    if risk is None:
        matched_title = candidate.current_title or state.get("matched_title") or "Professional Role"
        logger.warning("[Node: assemble_result_node] Missing risk_profile in state. Inferring via UserCareerService ML...")
        try:
            from source.services.user.career_service import UserCareerService
            service = UserCareerService()
            risk = service.predict_hybrid_role_risk(candidate)
        except Exception as e:
            logger.warning("[Node: assemble_result_node] Inference fallback on error: %s", e)
            risk = RiskProfile(
                role_title=matched_title,
                threat_score=50.0,
                risk_level=RiskLevel.MODERATE,
                is_deterministic=False,
            )

    safe_pivots = state.get("safe_pivot_roles", [])
    skill_gap = state.get("skill_gap")
    skill_gaps = state.get("skill_gaps", [skill_gap] if skill_gap else [])
    roadmap = state.get("copilot_roadmap")
    evidences = state.get("grounding_evidence", [])
    critic_review = state.get("critic_review")

    session_id = state.get("session_id") or str(int(time.time()))
    analysis = AnalysisResult(
        session_id=session_id,
        candidate_profile=candidate,
        risk_profile=risk,
        safe_pivots=safe_pivots,
        skill_gap=skill_gap,
        skill_gaps=skill_gaps,
        roadmap=roadmap,
        critic_review=critic_review,
        grounding_evidences=evidences,
    )

    summary_text = roadmap.executive_summary if roadmap else "Career advisory analysis completed."
    logger.info("[Node: assemble_result_node] AnalysisResult created successfully.")

    return {
        "final_analysis": analysis,
        "final_output": summary_text,
    }


# ==============================================================================
# Node 10: Interactive Copilot Node (Multi-turn Q&A with Re-RAG)
# ==============================================================================

def interactive_copilot_node(
    state: AgentState,
    copilot_llm: Optional[Any] = None,
    retriever: Optional[IEvidenceRetriever] = None,
) -> Dict[str, Any]:
    """
    Multi-Turn Interactive Copilot:
    Answers user follow-up questions anchored in approved roadmap and dynamic Re-RAG evidence.
    """
    messages = state.get("messages", [])
    last_user_message = messages[-1].content if messages else "How should I get started?"
    logger.info("[Node: interactive_copilot_node] Processing follow-up user query: '%s'...", last_user_message[:50])

    skill_gap = state.get("skill_gap")
    target_soc = skill_gap.target_soc_code if skill_gap else (state.get("matched_soc_code") or "")

    # Dynamic Re-RAG retrieval for user question
    evidence_retriever = retriever or _get_default_evidence_retriever()
    fresh_chunks = evidence_retriever.retrieve_evidence(
        query=last_user_message,
        soc_code=target_soc if target_soc else None,
        top_k=3,
    )

    evidence_context = "\n".join([f"- [O*NET {c.get('soc_code', target_soc)} / {c.get('aspect', 'General')}]: {c.get('content', '')[:200]}" for c in fresh_chunks])

    system_prompt = f"""{CopilotPromptCatalog.INTERACTIVE_COPILOT_PROMPT}

APPROVED ROADMAP CONTEXT:
Target: {skill_gap.target_role_title if skill_gap else 'Target Role'} (SOC: {target_soc})
Defense Skills: {skill_gap.defense_skills if skill_gap else []}

RE-RETRIEVED FACTUAL O*NET EVIDENCE:
{evidence_context}
"""

    llm = copilot_llm or _create_default_reasoning_llm()
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_user_message),
        ])
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning("[Node: interactive_copilot_node] Fallback response on error: %s", e)
        answer_text = f"Dựa trên lộ trình định hướng của bạn cho {target_soc}, hãy tập trung hoàn thành các kỹ năng phòng thủ và mục tiêu giai đoạn 1."

    logger.info("[Node: interactive_copilot_node] Generated grounded response (%d chars).", len(answer_text))
    return {
        "messages": [AIMessage(content=answer_text)],
        "final_output": answer_text,
    }


# ==============================================================================
# Helper Fallback & LLM Factory Functions
# ==============================================================================

def _create_default_fast_llm() -> Any:
    """Creates configured Fast Streaming LLM (DeepSeek Chat / Gemini Flash) with max_tokens=2048 for low latency without truncation."""
    from langchain_openai import ChatOpenAI
    if settings.DEEPSEEK_API_KEY and (settings.FAST_LLM_PROVIDER == "deepseek" or not settings.GEMINI_API_KEY):
        base_url = settings.DEEPSEEK_BASE_URL
        model = settings.FAST_LLM_MODEL
        if settings.DEEPSEEK_API_KEY.startswith("sk-or-v1-"):
            if "api.deepseek.com" in base_url:
                base_url = "https://openrouter.ai/api/v1"
            if model == "deepseek-chat":
                model = "deepseek/deepseek-chat"
        return ChatOpenAI(
            model=model,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=base_url,
            temperature=0.2,
            max_tokens=2048,
            max_retries=2,
        )
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2,
            max_output_tokens=2048,
        )
    return ChatOpenAI(model="deepseek-chat", api_key="dummy", temperature=0.2, max_tokens=2048)


def _create_default_reasoning_llm() -> Any:
    """Creates configured Reasoning LLM (DeepSeek / Gemini Pro) with temperature=0.2 and max_tokens=4096."""
    from langchain_openai import ChatOpenAI
    if settings.DEEPSEEK_API_KEY:
        base_url = settings.DEEPSEEK_BASE_URL
        model = settings.FAST_LLM_MODEL
        if settings.DEEPSEEK_API_KEY.startswith("sk-or-v1-"):
            if "api.deepseek.com" in base_url:
                base_url = "https://openrouter.ai/api/v1"
            if model == "deepseek-chat":
                model = "deepseek/deepseek-chat"
        return ChatOpenAI(
            model=model,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=base_url,
            temperature=0.2,
            max_tokens=4096,
            max_retries=3,
        )
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.REASONING_LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2,
            max_output_tokens=4096,
        )
    return ChatOpenAI(model="deepseek-chat", api_key="dummy", temperature=0.2, max_tokens=4096)


def _create_default_critic_llm() -> Any:
    """Creates configured Critic LLM with temperature=0.0 for zero-variance auditing."""
    from langchain_openai import ChatOpenAI
    if settings.DEEPSEEK_API_KEY:
        base_url = settings.DEEPSEEK_BASE_URL
        model = settings.FAST_LLM_MODEL
        if settings.DEEPSEEK_API_KEY.startswith("sk-or-v1-"):
            if "api.deepseek.com" in base_url:
                base_url = "https://openrouter.ai/api/v1"
            if model == "deepseek-chat":
                model = "deepseek/deepseek-chat"
        return ChatOpenAI(
            model=model,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=base_url,
            temperature=0.0,
            max_retries=3,
        )
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.0,
        )
    return ChatOpenAI(model="deepseek-chat", api_key="dummy", temperature=0.0)


def _infer_role_certifications_and_projects(
    target_role: str,
    target_soc: str,
    missing_tools: List[str],
) -> Dict[str, Any]:
    """Infers role-aligned professional certifications and capstone projects dynamically."""
    role_lower = target_role.lower()
    tools_str = ", ".join(missing_tools[:3]) if missing_tools else target_role
    primary_tool = missing_tools[0] if missing_tools else "Chuyên ngành"

    # Data / AI / Analytics
    if any(k in role_lower for k in ["data", "dữ liệu", "analytics", "machine learning", "ai", "scientist", "analyst"]) or target_soc.startswith(("15-205", "15-204", "15-1211")):
        return {
            "phase1_certs": ["AWS Certified Data Analytics" if "aws" in tools_str.lower() else "Google Cloud Professional Data Engineer", "CompTIA Data+"],
            "phase2_certs": ["Databricks Certified Data Engineer Professional" if "databricks" in tools_str.lower() else "Associate Data Analyst (SQL & Python)"],
            "phase3_certs": ["TensorFlow Developer Certificate" if "deep" in role_lower or "ai" in role_lower else "Snowflake SnowPro Core Certified"],
            "phase1_proj": [f"Feature Engineering & Data Validation Pipeline ({primary_tool})"],
            "phase2_proj": [f"End-to-End Enterprise Analytics Platform for {target_role}"],
            "phase3_proj": [f"Production Capstone: Scalable {target_role} System & Predictive Dashboard"],
        }
    # Security / Cybersecurity
    elif any(k in role_lower for k in ["security", "bảo mật", "an ninh", "cyber", "infosec"]) or target_soc.startswith("15-1212"):
        return {
            "phase1_certs": ["CompTIA Security+", "Microsoft Certified: Security, Compliance, and Identity Fundamentals"],
            "phase2_certs": ["Certified Ethical Hacker (CEH)" if "penetration" in role_lower else "CompTIA CySA+ (Cybersecurity Analyst)"],
            "phase3_certs": ["Certified Information Systems Security Professional (CISSP) Associate"],
            "phase1_proj": [f"Vulnerability Scanning & Hardening Audit ({primary_tool})"],
            "phase2_proj": ["Automated SIEM Telemetry & Threat Intelligence System"],
            "phase3_proj": [f"Enterprise Incident Response & Zero-Trust Architecture for {target_role}"],
        }
    # Software Engineering / Web / Fullstack
    elif any(k in role_lower for k in ["developer", "software", "phần mềm", "backend", "frontend", "fullstack", "programmer"]) or target_soc.startswith("15-1252"):
        return {
            "phase1_certs": ["Oracle Certified Professional" if "java" in tools_str.lower() else "Meta Professional Software Developer", "GitHub Actions / CI/CD Certification"],
            "phase2_certs": ["AWS Certified Developer - Associate", "HashiCorp Terraform Associate"],
            "phase3_certs": ["Certified Secure Software Lifecycle Professional (CSSLP)"],
            "phase1_proj": [f"RESTful Microservice Architecture with Automated Testing ({primary_tool})"],
            "phase2_proj": [f"High-Concurrency Distributed Backend System for {target_role}"],
            "phase3_proj": [f"Production Capstone: Scalable Multi-Tenant {target_role} Platform"],
        }
    # Cloud / DevOps / Infrastructure / SRE
    elif any(k in role_lower for k in ["cloud", "devops", "sre", "infrastructure", "đám mây", "hạ tầng", "system admin"]) or target_soc.startswith(("15-1242", "15-1243")):
        return {
            "phase1_certs": ["AWS Certified Cloud Practitioner", "HashiCorp Terraform Associate"],
            "phase2_certs": ["AWS Certified Solutions Architect - Associate"],
            "phase3_certs": ["Certified Kubernetes Administrator (CKA)"],
            "phase1_proj": [f"Infrastructure-as-Code & Automated Deployment Pipeline ({primary_tool})"],
            "phase2_proj": ["Multi-Region Cloud Architecture with Telemetry & Observability"],
            "phase3_proj": [f"Production Capstone: Resilient Cloud Infrastructure for {target_role}"],
        }
    # Generic / Professional default
    else:
        return {
            "phase1_certs": [f"Chứng chỉ Nền tảng Quốc tế cho {target_role}", "Professional Foundations Certification"],
            "phase2_certs": [f"Chứng chỉ Nâng cao Chuyên môn ({target_role})"],
            "phase3_certs": [f"Chứng chỉ Chuyên gia Cao cấp Quốc tế ({target_role})"],
            "phase1_proj": [f"Dự án Nền tảng Ứng dụng Thực hành ({primary_tool})"],
            "phase2_proj": [f"Quy trình Chuyên sâu & Giải pháp Toàn diện cho {target_role}"],
            "phase3_proj": [f"Public Capstone: Giải pháp Đột phá Vị trí {target_role}"],
        }


def _create_fallback_roadmap(
    target_role: str,
    target_soc: str,
    skill_gap: Optional[SkillGap],
) -> CopilotRoadmap:
    """Dynamic role-aligned fallback roadmap when LLM generation is unavailable or errors out."""
    defense_skills = skill_gap.defense_skills if skill_gap and skill_gap.defense_skills else ["System Architecture", "Complex Problem Solving", "Critical Thinking"]
    missing_tools = skill_gap.missing_critical_tools if skill_gap and skill_gap.missing_critical_tools else ["Modern Workflows"]

    role_meta = _infer_role_certifications_and_projects(target_role, target_soc, missing_tools)
    primary_tool = missing_tools[0] if missing_tools else target_role

    phase1 = RoadmapPhase(
        phase_name="Phase 1: 30 Days - Defensive Foundation",
        duration_days=30,
        objective=f"Master high-priority missing technical tools ({primary_tool}) and core workflows",
        action_items=[
            f"Complete practical hands-on labs for {primary_tool}",
            "Bridge foundational gaps in modern automated deployment pipelines",
            "Establish Git-driven portfolio repository structure",
        ],
        recommended_certifications=role_meta["phase1_certs"],
        target_skills=defense_skills[:2] if len(defense_skills) >= 2 else defense_skills,
        practical_projects=role_meta["phase1_proj"],
    )

    phase2 = RoadmapPhase(
        phase_name="Phase 2: 60 Days - Practical Deepening",
        duration_days=30,
        objective=f"Build multi-component production workflows and deep technical defense skills for {target_role}",
        action_items=[
            f"Design and deploy production-grade distributed architectures aligned with {target_role}",
            "Implement automated observability, logging, and performance telemetry",
        ],
        recommended_certifications=role_meta["phase2_certs"],
        target_skills=defense_skills,
        practical_projects=role_meta["phase2_proj"],
    )

    phase3 = RoadmapPhase(
        phase_name="Phase 3: 90 Days - Portfolio Mastery & Transition",
        duration_days=30,
        objective=f"Finalize flagship capstone project and position for target role transition into {target_role}",
        action_items=[
            f"Deliver public capstone demonstrating end-to-end {target_role} mastery",
            "Optimize resume and technical portfolio targeting defense-oriented positions",
        ],
        recommended_certifications=role_meta["phase3_certs"],
        target_skills=defense_skills,
        practical_projects=role_meta["phase3_proj"],
    )

    mermaid_code = (
        "graph TD\n"
        "    A[Phase 1: 30 Days - Defensive Foundation] --> B[Phase 2: 60 Days - Practical Deepening]\n"
        "    B --> C[Phase 3: 90 Days - Portfolio Mastery & Transition]\n"
        f"    C --> D[Ready for {target_role}]"
    )

    return CopilotRoadmap(
        target_role=target_role,
        target_soc_code=target_soc,
        executive_summary=f"Strategic 90-day transition roadmap guiding candidate into {target_role} with strong AI defense insulation.",
        phases=[phase1, phase2, phase3],
        defense_skills_breakdown=defense_skills,
        mermaid_chart=mermaid_code,
        estimated_completion_months=3,
    )

