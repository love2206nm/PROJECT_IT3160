# Module 5: Hierarchical Aspect-Based RAG Engine (Mỏ Neo Dẫn Chứng O*NET)

---

## 1. Mục Tiêu Module
Triển khai giao diện `IEvidenceRetriever` từ `source/domain/interfaces/rag.py` theo kiến trúc **Hierarchical Aspect-Based RAG Pattern**:
* **Không nhồi nhét văn bản cồng kềnh**: Mỗi nghề ($N=867$) được xé nhỏ thành đúng **3 Micro-Chunks theo khía cạnh độc lập** (`threat_mechanism`, `defense_skills`, `tasks_and_tools`), giải quyết triệt để hiện tượng pha loãng thông tin và "Lost in the Middle".
* **Lọc Namespace 2 tầng (Pre-filtering)**: Cho phép truy vấn khoanh vùng chính xác theo **Nhóm ngành O\*NET (`major_group` 2 chữ số)** và **Khía cạnh (`aspect`)**, triệt tiêu 100% rác ngữ nghĩa xuyên ngành.
* **Hybrid Search & Reranking**: Kết hợp **BM25Okapi** (bắt chính xác mã công cụ/kỹ năng) và **FAISS** (ngữ nghĩa embedding) trên tập đã lọc, sau đó dùng **Cross-Encoder Reranker** để chốt Top 3-5 dẫn chứng sự thật làm mỏ neo (Grounding Truth) cho Copilot Agent.

---

## 2. Danh Sách Task Cần Thực Hiện

### Task 5.1: Knowledge Corpus Management (`source/model/LLM_model/rag/corpus_manager.py`)
* **File triển khai**: `source/model/LLM_model/rag/corpus_manager.py`
* **Nguồn dữ liệu**:
  - `data/clean_data/final/rag_corpus.json` (được sinh sẵn từ Task 02 qua `OnetMetadataBuilder.build_rag_corpus()` gồm chính xác 2,601 micro-chunks, 23 major groups, 3 aspects: `threat_mechanism`, `defense_skills`, `tasks_and_tools`).
* **Nhiệm vụ cụ thể**:
  - Lập chỉ mục tra cứu ngược trong bộ nhớ: `_by_major_group`, `_by_aspect`, `_by_soc_code`.
  - Cung cấp phương thức `filter_chunks(major_group, aspect, soc_code)` thực hiện Pre-filtering Namespace $O(1)$.
  - Cung cấp `get_chunk_by_id(chunk_id)` và `all_chunks`.

---

### Task 5.2: Hierarchical Retriever Implementation (`source/model/LLM_model/rag/retriever.py`)
* **File cần code**: `source/model/LLM_model/rag/retriever.py`
* **Triển khai Interface**: `IEvidenceRetriever` từ `domain/interfaces/rag.py`.
* **Quy trình truy xuất 3 giai đoạn**:
  ```mermaid
  flowchart TD
      Query["Copilot Query (vd: Kỹ năng phòng thủ Data Analyst)"] --> Router["Metadata & Aspect Resolver"]
      Router -->|major_group='15', aspect='defense_skills'| Filter["Stage 1: Pre-filtering Namespace"]
      Corpus["2,601 Micro-Chunks"] --> Filter
      Filter -->|Filtered ~35 chunks| Hybrid["Stage 2: Hybrid Search (BM25 + FAISS)"]
      Hybrid -->|Top 10 candidates| Reranker["Stage 3: Cross-Encoder Reranker"]
      Reranker --> TopK["Top 3-5 Chunks Dẫn Chứng Tinh Khiết"]
  ```
  1. **Stage 1 (Pre-filtering)**: Lọc tức thì theo `major_group` (nếu biết) và `aspect` (dựa trên context của Node trong Copilot Graph). Nếu không truyền `major_group`, hỗ trợ chế độ tìm kiếm toàn cục (Global Fallback).
  2. **Stage 2 (Hybrid Search)**:
     - **BM25Okapi**: Bắt chính xác tên thư viện, tool, chứng chỉ (vd: `Docker`, `PyTorch`, `AWS`).
     - **FAISS (Dense Semantic)**: Sử dụng mô hình embedding nhỏ gọn (ví dụ `all-MiniLM-L6-v2`) để tính Cosine Similarity.
     - Kết hợp điểm số bằng phương pháp Reciprocal Rank Fusion (RRF) hoặc weighted sum ($0.5 \times \text{BM25} + 0.5 \times \text{FAISS}$).
  3. **Stage 3 (Cross-Encoder Reranking)**:
     - Sử dụng `cross-encoder/ms-marco-MiniLM-L-6-v2` để chấm điểm tương quan cặp (Query, Chunk Text), triệt tiêu false positives.
     - Trả về danh sách đối tượng `EvidenceChunk` chuẩn mực.

---

### Task 5.3: Hybrid Indexer & CLI Xây Dựng Vector Index (`models/faiss_index/`)
* **Files triển khai**: 
  - `source/model/LLM_model/rag/indexer.py`: Quản lý `BM25Okapi` sparse index và `faiss.IndexFlatIP` dense vector index.
  - `source/services/cli/build_rag_index_cli.py`: CLI chạy sinh và lưu index ra `models/faiss_index/` (`dense_faiss.index` 3.9 MB, `hybrid_meta.joblib` 1.3 MB).
  - Tách biệt 100% vector index sang thư mục `models/` (không lẫn vào `data/`).

---

### Task 5.4: Test Suite & API Safety (`tests/test_rag.py` & `tests/conftest.py`)
* **Files triển khai**: `tests/test_rag.py`, `tests/conftest.py`
* **Cơ chế an toàn API Token**:
  - `tests/conftest.py` đăng ký `--run-live-api` và marker `@pytest.mark.live_api`.
  - Mặc định khi chạy `pytest`: Toàn bộ các test gọi API ngoài sẽ bị `SKIPPED`, đảm bảo không tốn phí/token.
* **Kịch bản kiểm thử**:
  - `test_corpus_manager_initialization_and_metadata`: Nạp đủ 2,601 chunks, 23 major groups, 3 aspects.
  - `test_corpus_manager_filter_major_group`: Khoanh vùng O*NET major group (vd: 15 cho IT).
  - `test_corpus_manager_filter_aspect`: Khoanh vùng theo khía cạnh (vd: defense_skills).
  - `test_hybrid_indexer_sparse_and_dense_search`: Kiểm tra hoạt động của BM25 và FAISS.
  - `test_retriever_protocol_conformance`: Xác minh tuân thủ `IEvidenceRetriever`.
  - `test_retriever_prefiltering_and_aspect_isolation`: Kiểm tra độ chính xác lọc 2 tầng.
  - `test_retriever_global_fallback`: Truy vấn không có bộ lọc.
* **Lệnh chạy test**:
  ```powershell
  pytest tests/test_rag.py -v
  ```
