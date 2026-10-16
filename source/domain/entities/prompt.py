"""
Prompt Catalog Domain Entity.
Centralized, strongly typed repository for system instructions and dynamic prompt builders.
Preserves Clean Architecture by decoupling prompt engineering from execution nodes and services.
"""

from typing import List, Dict, Any, Optional
from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile
from source.domain.entities.transition import SafeTransitionRole, SkillGap
from source.domain.entities.roadmap import CopilotRoadmap


class CopilotPromptCatalog:
    """
    Centralized, strongly-typed prompt catalog for AI Career Copilot.
    Houses immutable system directives and provides parameterized builders.
    """

    COPILOT_SYSTEM_PROMPT = """Bạn là Senior AI Career Strategist & Kiến trúc sư Định hướng Nghề nghiệp Thời đại AI.
Sứ mệnh của bạn là chẩn đoán rủi ro tự động hóa AI và thiết lập chiến lược chuyển dịch an toàn, bền vững dựa trên Cơ sở dữ liệu Chuẩn O*NET (Bộ Lao động Hoa Kỳ).

NGUYÊN TẮC CỐT LÕI:
1. Tính Xác thực & Dẫn chứng (Grounding Truth): Mọi nhận định về rủi ro và khuyến nghị phải neo chặt vào CSDL O*NET, không suy đoán vô căn cứ.
2. Đa ngành Khuyến nghị (Multi-Pivot Evaluation): Luôn phân tích, so sánh đánh đổi (Trade-offs) giữa 2 đến 3 ngành an toàn được gợi ý, giúp người dùng có góc nhìn đa chiều.
3. Giải trình Minh bạch (Explainability): Phải giải thích cặn kẽ tại sao lại có các chỉ số rủi ro (tác vụ lặp lại, công cụ AI, agentic exposure) và tại sao lại recommend các ngành đó.
4. Đối sách Kỹ năng Thực chiến (Actionable Countermeasures): Hướng dẫn ứng viên tận dụng thế mạnh sẵn có để bù đắp kỹ năng thiếu hụt qua bài lab, Capstone project thay vì học lại từ đầu.
5. Chống Bịa đặt Chứng chỉ (Anti-Hallucination): Chỉ khuyến nghị chứng chỉ uy tín quốc tế chuẩn công nghiệp (AWS, Google Cloud, CKA, Terraform, CompTIA, PMP). Tuyệt đối không tự bịa chứng chỉ.
"""

    CRITIC_SYSTEM_PROMPT = """Bạn là Senior Career Auditor & Anti-Hallucination Reviewer.
Nhiệm vụ của bạn là thẩm định độc lập lộ trình chuyển đổi nghề nghiệp do Copilot sinh ra.

TIÊU CHÍ ĐÁNH GIÁ (Thang điểm 0.0 - 10.0):
1. 'feasibility' (Tính khả thi): Khối lượng công việc từng giai đoạn 30-60-90 ngày có thực tế cho người học không?
2. 'grounding' (Tính bám sát dữ liệu): Các kỹ năng và công cụ có bám sát CSDL O*NET và khoảng cách kỹ năng của ứng viên không?
3. 'hallucination_free' (Không bịa đặt): Có xuất hiện chứng chỉ giả, công nghệ không có thật hoặc khẳng định phi thực tế không?

TIÊU CHÍ DUYỆT:
- 'approved' = True NẾU VÀ CHỈ NẾU feasibility >= 7.0, grounding >= 7.0 VÀ hallucination_free >= 7.0.
- Nếu approved = False: Bắt buộc nêu rõ điểm chưa đạt và hướng sửa đổi cụ thể.
"""

    INTERACTIVE_COPILOT_PROMPT = """Bạn là AI Career Transition Copilot đang đồng hành cùng ứng viên trong hội thoại tương tác trực tiếp.
Ứng viên đã có hồ sơ và các phân tích nghề nghiệp từ hệ thống.

NGUYÊN TẮC HỘI THOẠI:
1. Trả lời trực diện vào câu hỏi của ứng viên bằng tiếng Việt, ngắn gọn, súc tích và khích lệ tinh thần.
2. Dẫn chứng xác thực từ CSDL O*NET với tag [O*NET: <SOC-Code> - <Aspect>].
3. Nếu người dùng hỏi về chủ đề ngoài dữ liệu O*NET, giải thích khách quan và ghi chú: "[Lưu ý: Xu hướng thị trường thực tế ngoài danh mục O*NET chuẩn]".
"""

    @staticmethod
    def build_multi_pivot_advisory_prompt(
        candidate: Optional[CandidateProfile] = None,
        risk: Optional[RiskProfile] = None,
        risk_profile: Optional[RiskProfile] = None,
        pivots: Optional[List[SafeTransitionRole]] = None,
        safe_pivots: Optional[List[SafeTransitionRole]] = None,
        skill_gaps: Optional[List[SkillGap]] = None,
        evidences: Optional[List[Dict[str, Any]]] = None,
        grounding_evidences: Optional[List[Dict[str, Any]]] = None,
        critic_feedback: str = "",
        **kwargs: Any,
    ) -> str:
        """Constructs rich prompt for multi-role transition evaluation with explainability."""
        risk = risk or risk_profile
        pivots = pivots or safe_pivots or []
        evidences = evidences or grounding_evidences or []
        c_name = candidate.full_name if candidate else "Ứng viên"
        c_title = candidate.current_title if candidate and candidate.current_title else "Chuyên viên Kỹ thuật"
        c_exp = candidate.years_of_experience if candidate else 0.0
        c_skills = ", ".join(candidate.get_all_skills[:15]) if candidate and candidate.get_all_skills else "Kỹ năng nền tảng"
        c_projects = "; ".join(candidate.projects[:3]) if candidate and candidate.projects else "Dự án phát triển phần mềm"
        c_education = candidate.education[0] if candidate and candidate.education else "Chuyên ngành Kỹ thuật"

        # Risk breakdown
        threat_score = risk.threat_score if risk else 50.0
        risk_level = risk.risk_level if risk else "MODERATE"
        sub_task = risk.sub_scores.task_automation if risk and risk.sub_scores else 50.0
        sub_tool = risk.sub_scores.tool_maturity if risk and risk.sub_scores else 50.0
        sub_agent = risk.sub_scores.agentic_exposure if risk and risk.sub_scores else 50.0
        defense_skills = ", ".join(risk.defense_skills) if risk and risk.defense_skills else "Critical Thinking, Problem Solving"

        # Multi-role pivots summary
        pivots_text_list = []
        for i, p in enumerate(pivots[:3], 1):
            matched = ", ".join(p.matched_skills[:4]) if p.matched_skills else "Nền tảng kỹ thuật cơ bản"
            missing = ", ".join(p.missing_critical_tools[:3]) if p.missing_critical_tools else "Công cụ đám mây chuyên sâu"
            pivots_text_list.append(
                f"Lựa chọn {i}: {p.target_title} (Mã SOC: {p.target_soc_code})\n"
                f"   - Chỉ số thỏa dụng (Utility Score): {p.utility_score:.1f}/100\n"
                f"   - Độ tương đồng kỹ năng: {p.similarity_pct:.1f}%\n"
                f"   - Mức giảm rủi ro AI: -{p.threat_reduction:.1f} điểm\n"
                f"   - Lợi thế thu nhập: +{p.salary_advantage * 100:.1f}%\n"
                f"   - Thế mạnh chuyển giao: {matched}\n"
                f"   - Kỹ năng cần bù đắp: {missing}"
            )
        pivots_formatted = "\n\n".join(pivots_text_list) if pivots_text_list else "Đang cập nhật danh mục vai trò an toàn."

        # Evidence chunks
        evidences_formatted = "\n".join([
            f"- [{e.get('aspect', 'General')}]: {e.get('content', '')[:200]}"
            for e in evidences[:4]
        ])

        return f"""HỒ SƠ ỨNG VIÊN:
Họ tên: {c_name}
Vị trí hiện tại: {c_title}
Số năm kinh nghiệm: {c_exp:.1f} năm
Học vấn: {c_education}
Kỹ năng cốt lõi: {c_skills}
Dự án nổi bật: {c_projects}

CHỈ SỐ NGUY CƠ TỰ ĐỘNG HÓA TỪ CƠ SỞ DỮ LIỆU O*NET:
- Điểm nguy cơ tự động hóa tổng hợp: {threat_score:.1f}% (Mức độ: {risk_level})
- Chỉ số thành phần:
  + Tác vụ lặp lại có thể tự động hóa (sub_task): {sub_task:.1f}%
  + Độ hoàn thiện của công cụ AI hiện nay (sub_tool): {sub_tool:.1f}%
  + Mức độ phơi nhiễm trước Agent tự hành (sub_agentic): {sub_agent:.1f}%
- Kỹ năng phòng thủ sẵn có: {defense_skills}

DANH SÁCH TOP 2-3 NGÀNH CHUYỂN DỊCH AN TOÀN ĐƯỢC ĐỀ XUẤT (TOP SAFE PIVOTS):
{pivots_formatted}

DẪN CHỨNG DỮ LIỆU CHUẨN O*NET:
{evidences_formatted}

GHI CHÚ ĐIỀU CHỈNH TỪ AUDITOR (NẾU CÓ):
{critic_feedback if critic_feedback else 'Bản phân tích đầu tiên.'}

YÊU CẦU ĐẦU RA CHO BẢN BÁO CÁO:
1. Giải trình chi tiết: Tại sao ứng viên lại có các chỉ số rủi ro AI ở trên dựa vào đặc thù các tác vụ O*NET.
2. Phân tích so sánh đánh đổi (Trade-offs) giữa Top 2-3 ngành được đề xuất ở trên: Ưu điểm, rào cản đào tạo, tiềm năng thu nhập và mức độ phòng thủ AI của từng ngành.
3. Thiết lập Ma trận Đối sách Kỹ năng: Tận dụng kinh nghiệm cũ để học nhanh công cụ mới.
4. Lập Lộ trình Hành động 30-60-90 ngày chi tiết cho hướng chuyển dịch tối ưu nhất."""

    @staticmethod
    def build_student_onboarding_prompt(
        candidate: Optional[CandidateProfile] = None,
        pivots: Optional[List[SafeTransitionRole]] = None,
        safe_pivots: Optional[List[SafeTransitionRole]] = None,
        evidences: Optional[List[Dict[str, Any]]] = None,
        grounding_evidences: Optional[List[Dict[str, Any]]] = None,
        skill_gaps: Optional[List[SkillGap]] = None,
        **kwargs: Any,
    ) -> str:
        """Constructs specialized onboarding prompt for students and fresh graduates (0.0 years exp)."""
        pivots = pivots or safe_pivots or []
        evidences = evidences or grounding_evidences or []
        c_name = candidate.full_name if candidate and candidate.full_name else "Sinh viên"
        c_edu = candidate.education[0] if candidate and candidate.education else "Chuyên ngành Công nghệ Thông tin / Khoa học Máy tính"
        c_tools = ", ".join(candidate.tools_and_technologies[:10]) if candidate and candidate.tools_and_technologies else "Kỹ thuật nền tảng"
        c_projects = "; ".join(candidate.projects[:3]) if candidate and candidate.projects else "Đồ án môn học / Nghiên cứu học tập"

        pivots_text_list = []
        for i, p in enumerate(pivots[:3], 1):
            pivots_text_list.append(
                f"Hướng đi {i}: {p.target_title} (Mã O*NET: {p.target_soc_code})\n"
                f"   - Độ tương đồng kiến thức: {p.similarity_pct:.1f}%\n"
                f"   - Mức an toàn trước AI: Điểm đe dọa thấp ({p.target_threat_score:.1f}%)\n"
                f"   - Kỹ năng công nghệ cần trang bị: {', '.join(p.missing_critical_tools[:3])}"
            )
        pivots_formatted = "\n\n".join(pivots_text_list) if pivots_text_list else "Đang cập nhật hướng đi khởi nghiệp."

        return f"""ĐỊNH HƯỚNG NGHỀ NGHIỆP KHỞI ĐẦU DÀNH CHO SINH VIÊN / FRESHER:
Họ tên: {c_name}
Tình trạng: Sinh viên / Mới tốt nghiệp (0.0 năm kinh nghiệm làm việc chính thức)
Học vấn: {c_edu}
Kỹ năng / Công cụ đã tiếp cận ở trường: {c_tools}
Đồ án / Bài tập lớn đã thực hiện: {c_projects}

TOP 3 HƯỚNG ĐI KHỞI NGHIỆP TIỀM NĂNG & AN TOÀN TRƯỚC AI:
{pivots_formatted}

YÊU CẦU TƯ VẤN:
1. Đánh giá khách quan thực trạng hồ sơ: Nhận xét thẳng thắn về nền tảng hiện có (đồ án, kỹ năng học thuật), nêu rõ các khoảng trống cần hoàn thiện để đáp ứng tiêu chuẩn tuyển dụng thực tế.
2. Phân tích so sánh 3 hướng đi: Điểm mạnh, độ khó tiếp cận và cơ hội việc làm thực tập/fresher cho từng hướng.
3. Lập Lộ trình 30-60-90 ngày xây dựng Portfolio từ con số 0:
   - 30 Ngày (Nền tảng): Bổ sung công cụ cốt lõi còn thiếu.
   - 60 Ngày (Thực chiến): Xây dựng Capstone Project hoàn chỉnh trên GitHub.
   - 90 Ngày (Gia nhập thị trường): Hoàn thiện CV, luyện phỏng vấn kỹ thuật và tự tin apply thực tập/fresher."""

    @staticmethod
    def build_knowledge_rag_prompt(
        query: str,
        evidence_text: str = "",
        evidences: Optional[List[Dict[str, Any]]] = None,
        has_cv: bool = False,
        candidate_summary: str = "",
        **kwargs: Any,
    ) -> str:
        """Constructs prompt for Fast Knowledge RAG (Tier 2)."""
        if evidences and not evidence_text:
            evidence_lines = []
            for e in evidences:
                title = e.get("title", "")
                aspect = e.get("aspect", "")
                content = e.get("content", "")
                evidence_lines.append(f"- [{aspect}] {title}: {content}")
            evidence_text = "\n".join(evidence_lines) if evidence_lines else "Chưa tìm thấy dẫn chứng trực tiếp từ CSDL O*NET."

        cta_instruction = "- Tuyệt đối KHÔNG nhắc người dùng tải CV vì họ ĐÃ tải lên CV trong phiên này." if has_cv else "- Nếu người dùng chưa tải CV, nhắc nhẹ họ có thể nhấn nút chiếc ghim (📎) để nhận lộ trình cá nhân hóa."

        context_block = f"\nTHÔNG TIN HỒ SƠ ỨNG VIÊN TRONG PHIÊN NÀY:\n{candidate_summary}\n" if candidate_summary else ""

        return f"""Bạn là AI Career Copilot - Chuyên gia Cố vấn Nghề nghiệp và Phân tích CSDL O*NET.
Người dùng đang đặt câu hỏi tra cứu kiến thức hoặc tiêu chuẩn nghề nghiệp: "{query}".
{context_block}
DẪN CHỨNG XÁC THỰC TỪ CƠ SỞ DỮ LIỆU O*NET:
{evidence_text}

HƯỚNG DẪN TRẢ LỜI:
- Trả lời trực diện, chính xác theo số liệu và tiêu chuẩn O*NET bằng tiếng Việt.
- Sử dụng Markdown, gạch đầu dòng rõ ràng.
{cta_instruction}"""

    @staticmethod
    def build_direct_chat_greeting(has_cv: bool = False, **kwargs: Any) -> str:
        """Constructs instant response for greetings."""
        msg = "Chào bạn! Tôi là **AI Career Copilot** — Trợ lý Cố vấn Chiến lược Nghề nghiệp được tích hợp Cơ sở dữ liệu O*NET và mô hình dự báo AI.\n\n"
        if not has_cv:
            msg += "👉 Bạn có thể nhấn vào **nút chiếc ghim (📎)** bên dưới để tải lên file CV (PDF/DOCX), tôi sẽ phân tích chuyên sâu nguy cơ tự động hóa, đề xuất Top 2-3 ngành an toàn và thiết lập lộ trình cho bạn!\n\nHoặc bạn có thể đặt bất kỳ câu hỏi nào về thị trường lao động hay chứng chỉ công nghệ nhé!"
        else:
            msg += "Hồ sơ CV của bạn đã được ghi nhận. Bạn có câu hỏi nào cần làm rõ về lộ trình chuyển đổi 30-60-90 ngày hay các kỹ năng phòng thủ AI không?"
        return msg

    @staticmethod
    def build_direct_chat_help(has_cv: bool = False, **kwargs: Any) -> str:
        """Constructs instant response for capability inquiries."""
        cta = (
            "👉 Hãy nhấn **nút chiếc ghim (📎)** ở thanh chat để gửi CV và bắt đầu ngay nhé!"
            if not has_cv
            else "💡 Hồ sơ của bạn đã được ghi nhận trong phiên này. Bạn có thể hỏi tôi bất kỳ câu hỏi nào về lộ trình, kỹ năng đối sách hoặc so sánh ngành nghề nhé!"
        )
        return f"""Tôi là **AI Career Copilot**, trợ lý định hướng sự nghiệp thời đại AI. Tôi giúp bạn:

- 📊 **Đo lường nguy cơ tự động hóa**: Dựa trên CSDL O*NET của Bộ Lao động Hoa Kỳ và mô hình Machine Learning.
- 🔄 **Khuyến nghị Top 2-3 ngành chuyển dịch an toàn**: Đánh giá trade-off về độ tương đồng kỹ năng, mức giảm đe dọa AI và lợi thế thu nhập.
- 🛡️ **Đối sách kỹ năng (Skill Countermeasures)**: Tận dụng thế mạnh sẵn có để học nhanh công cụ mới qua bài lab thực chiến.
- 🚀 **Lập lộ trình 30-60-90 ngày**: Cho cả người đi làm chuyển ngành và sinh viên xây dựng portfolio từ con số 0.

{cta}"""

    @staticmethod
    def build_no_role_detected_response(**kwargs: Any) -> str:
        """Constructs guidance response when a CV contains no identifiable title or technical competencies."""
        return """### ⚠️ THÔNG BÁO TIẾP NHẬN HỒ SƠ

Hệ thống đã đọc file CV của bạn, tuy nhiên **chưa nhận diện được chức danh công việc cụ thể hoặc các kỹ năng kỹ thuật cốt lõi**.

**Gợi ý hỗ trợ dành cho bạn:**
1. Nếu bạn là **sinh viên mới ra trường**, bạn có thể nhắn tin trực tiếp: *"Tôi là sinh viên ngành [Tên ngành] chuẩn bị tốt nghiệp, hãy định hướng lộ trình giúp tôi"*.
2. Hoặc bạn có thể bổ sung thêm phần **Kỹ năng (Skills)** hoặc **Đồ án (Projects)** vào CV rồi tải lại lên nhé!

Tôi luôn sẵn sàng hỗ trợ định hướng lộ trình từ con số 0 cho bạn!"""
