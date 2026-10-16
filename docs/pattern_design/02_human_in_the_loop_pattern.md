# Pattern 02: Human-in-the-Loop (HITL) State Machine Pattern

---

## 1. Bối Cảnh & Bài Toán Giải Quyết
Khi ứng viên tải CV lên, các thuật toán trích xuất dù dùng LLM mạnh đến đâu vẫn có thể gặp rủi ro:
* Đọc nhầm chức danh (ví dụ: nhầm chức danh trong một dự án cũ thành chức danh chính hiện tại).
* Bỏ sót các kỹ năng cốt lõi hoặc nhận diện sai tên viết tắt công nghệ.

Nếu để các lỗi này truyền tiếp vào các tầng phân cụm, tính điểm rủi ro và sinh lộ trình, hệ thống sẽ đưa ra kết quả sai lệch hoàn toàn (Garbage In, Garbage Out).

**Giải pháp Human-in-the-Loop Pattern**:
* Đặt một điểm ngắt trạng thái (**Interrupt Breakpoint**) ngay sau bước bóc tách CV.
* Hệ thống tạm dừng thực thi, bảo lưu trạng thái phân tích vào bộ nhớ (`checkpointer`).
* Gửi `CandidateProfile` lên giao diện để người dùng kiểm tra nhanh trong 3 giây (xác nhận hoặc sửa đổi chức danh, bổ sung kỹ năng).
* Chỉ khi người dùng bấm xác nhận, hệ thống mới nhận tín hiệu `resume` và chạy tiếp các tầng ML/RAG/Copilot.

---

## 2. Sơ Đồ Trạng Thái & Điểm Ngắt (State Breakpoint Diagram)

```mermaid
stateDiagram-v2
    [*] --> cv_ingest_node
    cv_ingest_node --> extract_profile_node
    extract_profile_node --> human_review_node: Tự động lưu Checkpoint

    state human_review_node {
        [*] --> PAUSED: Ngắt thực thi (interrupt_before)
        PAUSED --> WAITING_USER: Đẩy Profile lên Frontend
        WAITING_USER --> RESUMED: Người dùng gửi POST /confirm
    }

    human_review_node --> router_node: Chạy tiếp sau khi Resume
    router_node --> [*]
```

---

## 3. Sơ Đồ Tuần Tự (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    actor User as Người dùng
    participant UI as Giao diện Web
    participant API as FastAPI Backend
    participant Graph as LangGraph Engine
    participant Checkpoint as MemorySaver DB

    User->>UI: Upload file CV (PDF/DOCX)
    UI->>API: POST /api/v1/cv/upload
    API->>Graph: app.invoke(inputs, config={"configurable": {"thread_id": "session-123"}})
    Graph->>Graph: Ingest & Extract Profile
    Note over Graph: Dừng lại trước human_review_node!
    Graph->>Checkpoint: Lưu toàn bộ state hiện tại
    Graph-->>API: Trả về trạng thái PAUSED + candidate_profile
    API-->>UI: Hiển thị Pop-up kiểm tra hồ sơ (3s)

    User->>UI: Xác nhận hoặc sửa chức danh/kỹ năng
    UI->>API: POST /api/v1/profile/confirm (thread_id="session-123", updated_profile)
    API->>Graph: app.update_state(config, {"candidate_profile": updated_profile, "user_confirmed": True})
    API->>Graph: app.invoke(None, config)  # Tiếp tục chạy tiếp
    Graph->>Graph: Chạy Router -> ML -> RAG -> Copilot
    Graph-->>API: Hoàn tất lộ trình
    API-->>UI: Hiển thị báo cáo và lộ trình học tập
```

---

## 4. Contract Interface

```python
from typing import TypedDict, Optional, Dict, Any

class HITLState(TypedDict):
    raw_cv_text: str
    candidate_profile: Optional[Dict[str, Any]]
    user_confirmed: bool  # True nếu người dùng đã xác thực
    thread_id: str
```

---

## 5. Khung Code Mẫu (Skeleton Boilerplate)

```python
# source/model/LLM_model/copilot/graph_hitl.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from source.model.LLM_model.copilot.state import AgentState

def cv_ingest_node(state: AgentState):
    # TODO: Đọc text blocks từ file
    return {"raw_cv_text": "sample text"}

def extract_profile_node(state: AgentState):
    # TODO: Gọi Fast LLM trích xuất JSON
    return {"candidate_profile": {"current_title": "Backend Developer", "skills": ["Python"]}}

def human_review_node(state: AgentState):
    """
    Điểm dừng tương tác.
    Khi graph bị ngắt trước node này, hệ thống sẽ dừng chờ.
    Khi resume, hàm này kiểm tra cờ user_confirmed.
    """
    print(f"Xác nhận hồ sơ cho chức danh: {state['candidate_profile']['current_title']}")
    return {"user_confirmed": True}

# Xây dựng đồ thị
workflow = StateGraph(AgentState)
workflow.add_node("cv_ingest_node", cv_ingest_node)
workflow.add_node("extract_profile_node", extract_profile_node)
workflow.add_node("human_review_node", human_review_node)

workflow.add_edge(START, "cv_ingest_node")
workflow.add_edge("cv_ingest_node", "extract_profile_node")
workflow.add_edge("extract_profile_node", "human_review_node")

# Cấu hình ngắt trước human_review_node
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_review_node"]
)
```
