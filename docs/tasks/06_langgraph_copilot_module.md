# Module 6: LangGraph Copilot Agent (Hệ Thống Tác Tử Cố Vấn & Tự Phản Biện)

---

## 1. Mục Tiêu Module
Triển khai hệ thống tác tử đa bước bằng **LangGraph**:
* Sử dụng Domain Entities (`CandidateProfile`, `RiskProfile`, `SafeTransitionRole`, `CopilotRoadmap`) trong `AgentState`.
* Điểm ngắt tương tác người dùng **Human-in-the-Loop (`interrupt_before=["human_review_node"]`)**.
* Vòng lặp tự phản biện **Critic Evaluator Loop** kiểm định tính khả thi và loại bỏ ảo giác chứng chỉ trước khi hoàn tất.

---

## 2. Danh Sách Task Cần Code

### Task 6.1: Khai báo State Schema (`source/model/LLM_model/copilot/state.py`)
* **File cần code**: `source/model/LLM_model/copilot/state.py`
* **Nhiệm vụ**: Định nghĩa `AgentState` sử dụng các entities từ `domain`:
  ```python
  from source.domain.entities.candidate import CandidateProfile
  from source.domain.entities.risk import RiskProfile
  from source.domain.entities.transition import SafeTransitionRole, SkillGap
  from source.domain.entities.roadmap import CopilotRoadmap

  class AgentState(TypedDict):
      raw_cv_text: str
      candidate_profile: Optional[CandidateProfile]
      user_confirmed: bool
      risk_profile: Optional[RiskProfile]
      safe_pivot_roles: Optional[List[SafeTransitionRole]]
      skill_gap: Optional[SkillGap]
      grounding_evidence: Optional[List[Dict[str, Any]]]
      copilot_roadmap: Optional[CopilotRoadmap]
      critic_approved: bool
      critic_feedback: str
      iteration_count: int
  ```

---

### Task 6.2: Triển khai Các Nodes & StateGraph (`source/model/LLM_model/copilot/graph.py`)
* **File cần code**:
  - `source/model/LLM_model/copilot/nodes.py`: Chứa các hàm node nhận `state` và trả về cập nhật `state`.
  - `source/model/LLM_model/copilot/prompts.py`: System prompt cho Copilot (Senior Career Strategist) và Critic Agent.
  - `source/model/LLM_model/copilot/graph.py`: Lắp ráp StateGraph với Checkpointer `MemorySaver`.

---

### Task 6.3: Integration Test cho LangGraph (`tests/test_graph.py`)
* **File cần code**: `tests/test_graph.py`
* **Lệnh chạy test**:
  ```powershell
  pytest tests/test_graph.py -v
  ```
