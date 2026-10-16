# Cẩm Nang Thiết Kế Kiến Trúc & Design Patterns
## AI Recruiter & Career Copilot

Thư mục này chứa đặc tả kỹ thuật chuyên sâu về **6 Design Patterns** cốt lõi của hệ thống theo triết lý **Clean Architecture** (Robert C. Martin) và **Ports & Adapters (Hexagonal Architecture)**. 

---

## 1. Bản Đồ Mối Quan Hệ Giữa Các Design Patterns

```mermaid
flowchart TD
    subgraph PRESENTATION["Presentation Layer"]
        API["FastAPI Controller"]
    end

    subgraph APPLICATION["Application Layer (Use Cases)"]
        ORCHESTRATOR["Pattern 06: Clean Architecture Orchestrator (Services)"]
        ROUTER["Pattern 01: Hybrid Risk Router Strategy"]
    end

    subgraph DOMAIN_CORE["Domain Layer (Entities & Protocols)"]
        ENTITIES["Domain Entities (Candidate, Onet, Risk, Roadmap)"]
        PORTS["Domain Interfaces / Ports (IParser, IRepo, IML, IRAG)"]
    end

    subgraph ADAPTERS["Adapters & Infrastructure Layer"]
        P3["Pattern 03: Two-Tier LLM Architecture (Fast Extractor & Reasoning Copilot)"]
        P2["Pattern 02: Human-in-the-Loop Pattern (State Pause & Resume)"]
        P4["Pattern 04: Hierarchical Aspect-Based RAG Pattern (Hierarchical Truth Anchor)"]
        P5["Pattern 05: Stateful Multi-Agent Graph (LangGraph State Machine)"]
    end

    API --> ORCHESTRATOR
    ORCHESTRATOR --> ROUTER
    ORCHESTRATOR --> PORTS
    ROUTER --> PORTS
    ADAPTERS ..|> PORTS
    ADAPTERS --> ENTITIES
```

---

## 2. Danh Mục 6 Architectural Patterns Chi Tiết

| Pattern | Tên Pattern | File Tài Liệu | Vấn đề Giải Quyết | Vị trí Áp Dụng |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Hybrid Router Strategy** | [`01_hybrid_router_pattern.md`](file:///c:/Users/Hai%20Nam/full_project/Job_advisor/docs/pattern_design/01_hybrid_router_pattern.md) | Phân luồng giữa `IOnetRepository` (80% tra cứu 0ms) và `IRiskPredictor` (20% ML XGBoost). | `source/services/risk_router.py` |
| **02** | **Human-in-the-Loop (HITL)** | [`02_human_in_the_loop_pattern.md`](file:///c:/Users/Hai%20Nam/full_project/Job_advisor/docs/pattern_design/02_human_in_the_loop_pattern.md) | Điểm ngắt trạng thái (breakpoint) lưu checkpointer để người dùng kiểm chứng hồ sơ trong 3s. | `source/services/hitl_service.py` |
| **03** | **Two-Tier LLM Architecture** | [`03_two_tier_llm_pattern.md`](file:///c:/Users/Hai%20Nam/full_project/Job_advisor/docs/pattern_design/03_two_tier_llm_pattern.md) | Tách biệt Adapter: Fast LLM giá rẻ cho Parser, Advanced Reasoning LLM cho Copilot Agent. | `source/parsers/`, `source/model/LLM_model/` |
| **04** | **Hierarchical Aspect-Based RAG** | [`04_evidence_grounding_rag_pattern.md`](file:///c:/Users/Hai%20Nam/full_project/Job_advisor/docs/pattern_design/04_evidence_grounding_rag_pattern.md) | Mỏ neo sự thật đa tầng theo khía cạnh: Lọc Major Group O*NET + Aspect rồi Hybrid Search (FAISS + BM25) + Cross-Encoder. | `source/model/LLM_model/rag/` |
| **05** | **LangGraph State Machine** | [`05_langgraph_state_machine_pattern.md`](file:///c:/Users/Hai%20Nam/full_project/Job_advisor/docs/pattern_design/05_langgraph_state_machine_pattern.md) | Máy trạng thái đa tác tử có hướng, quản lý `AgentState` chứa các thực thể Domain. | `source/model/LLM_model/copilot/` |
| **06** | **Clean Use Case Orchestrator** | [`06_clean_architecture_orchestrator_pattern.md`](file:///c:/Users/Hai%20Nam/full_project/Job_advisor/docs/pattern_design/06_clean_architecture_orchestrator_pattern.md) | Quy định chuẩn: Service chỉ điều phối gọi hàm, Data Pipeline xử lý dữ liệu, Domain chứa Entity. | `source/services/cv_analysis_service.py` |
