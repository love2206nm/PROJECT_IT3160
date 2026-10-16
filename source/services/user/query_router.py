"""
Query Intent Router Module.
Implements a 3-Tier Query Dispatcher to optimize latency and decouple query handling:
- Tier 1 (DIRECT_CHAT): Fast conversational chit-chat, greetings, identity, and guidance (< 200ms).
- Tier 2 (KNOWLEDGE_RAG): Factual O*NET benchmark Q&A, roles, salary, and skill queries (< 800ms) without LangGraph overhead.
- Tier 3 (WORKFLOW): Stateful resume ingestion, deep career analysis, HITL overrides, and Critic reflection loops via ICopilotWorkflow.
"""

import re
from enum import Enum
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class QueryIntent(str, Enum):
    """Classification of user query intent."""
    DIRECT_CHAT = "direct_chat"
    KNOWLEDGE_RAG = "knowledge_rag"
    WORKFLOW = "workflow"


class QueryIntentRouter:
    """
    Production-grade Intent Dispatcher.
    Routes user messages to the most lightweight and appropriate processing pipeline.
    """

    DIRECT_CHAT_PATTERNS = [
        r"^(alo|alô|hi|hello|helo|chào|xin chào|hey|test)\b",
        r"\b(bạn là ai|bạn tên gì|giới thiệu|who are you|what can you do|bạn giúp gì)\b",
        r"\b(cảm ơn|thank|thanks|ok|okay|tuyệt vời|tạm biệt|bye)\b",
        r"\b(hướng dẫn|cách dùng|sử dụng như thế nào|tải cv|nút ghim|upload cv)\b",
    ]

    KNOWLEDGE_RAG_PATTERNS = [
        r"\b(soc|mã soc|mã nghề|onet|o\*net)\b",
        r"\b(lương|thu nhập|salary|compensation|usd|usd/năm)\b",
        r"\b(chứng chỉ|certificate|certification|cka|ckad|aws|azure|gcp|pmp|cissp)\b",
        r"\b(kỹ năng|skills?|công cụ|tools?|yêu cầu công việc|job description)\b",
        r"\b(nghề gì|vị trí gì|cloud architect|data engineer|devops|data scientist|software developer)\b",
        r"\b(thị trường|xu hướng|tương lai|tự động hóa|ai thay thế|nguy cơ)\b",
    ]

    WORKFLOW_PATTERNS = [
        r"\b(phân tích cv|đánh giá hồ sơ|phân tích hồ sơ|bóc tách cv)\b",
        r"\b(lập lộ trình|kế hoạch chuyển đổi|30-60-90|lộ trình 90 ngày)\b",
        r"\b(xác nhận|confirm|override|thay đổi mục tiêu|đổi chức danh)\b",
        r"\b(thẩm định|critic|đánh giá lại lộ trình|tối ưu phase)\b",
    ]

    CONTEXTUAL_WORKFLOW_PATTERNS = [
        r"\b(của tôi|hồ sơ của tôi|cv của tôi|điểm của tôi|nguy cơ của tôi|lộ trình của tôi)\b",
        r"\b(tôi nên chọn|hướng nào|ngành nào trong|lựa chọn nào|so sánh giữa|so sánh)\b",
        r"\b(tôi còn thiếu|bù đắp|kỹ năng của tôi|thế mạnh của tôi|tận dụng)\b",
        r"\b(30 ngày|giai đoạn 1|phase 1|chứng chỉ nào cho tôi|hướng thứ nhất|hướng thứ hai)\b",
    ]

    def route(
        self,
        message: str = "",
        has_file: bool = False,
        session_has_cv: bool = False,
    ) -> QueryIntent:
        """
        Dispatches incoming user interaction to the optimal Tier.

        Args:
            message: User prompt string
            has_file: Whether a CV file attachment is present
            session_has_cv: Whether current session already has an analyzed candidate CV

        Returns:
            QueryIntent: DIRECT_CHAT, KNOWLEDGE_RAG, or WORKFLOW
        """
        # Rule 1: Any interaction with a file attachment is strictly a WORKFLOW intent
        if has_file:
            logger.info("[QueryRouter] Detected file attachment -> Route: WORKFLOW")
            return QueryIntent.WORKFLOW

        msg_clean = message.strip().lower() if message else ""
        if not msg_clean:
            return QueryIntent.DIRECT_CHAT

        # Rule 2: Explicit workflow commands
        for pat in self.WORKFLOW_PATTERNS:
            if re.search(pat, msg_clean):
                logger.info("[QueryRouter] Matched workflow pattern '%s' -> Route: WORKFLOW", pat)
                return QueryIntent.WORKFLOW

        # Rule 3: Contextual workflow queries when session already has an analyzed CV
        if session_has_cv:
            for pat in self.CONTEXTUAL_WORKFLOW_PATTERNS:
                if re.search(pat, msg_clean):
                    logger.info("[QueryRouter] Matched contextual workflow pattern '%s' (has_cv=True) -> Route: WORKFLOW", pat)
                    return QueryIntent.WORKFLOW

        # Rule 4: Direct conversational greetings & guidance
        for pat in self.DIRECT_CHAT_PATTERNS:
            if re.search(pat, msg_clean):
                logger.info("[QueryRouter] Matched direct chat pattern '%s' -> Route: DIRECT_CHAT", pat)
                return QueryIntent.DIRECT_CHAT

        # Rule 5: Factual Knowledge & O*NET queries
        for pat in self.KNOWLEDGE_RAG_PATTERNS:
            if re.search(pat, msg_clean):
                logger.info("[QueryRouter] Matched knowledge RAG pattern '%s' -> Route: KNOWLEDGE_RAG", pat)
                return QueryIntent.KNOWLEDGE_RAG

        # Rule 6: Fallback based on session context
        if session_has_cv:
            logger.info("[QueryRouter] Active CV session fallback -> Route: KNOWLEDGE_RAG")
            return QueryIntent.KNOWLEDGE_RAG

        logger.info("[QueryRouter] Default fallback -> Route: DIRECT_CHAT")
        return QueryIntent.DIRECT_CHAT

