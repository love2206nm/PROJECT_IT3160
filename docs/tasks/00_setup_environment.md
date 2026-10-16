# Module 0: Môi Trường Phát Triển & Cấu Hình Hệ Thống

---

## 1. Mục Tiêu Module
Thiết lập môi trường Python ảo (`venv`), khai báo đầy đủ các thư viện phụ thuộc (`requirements.txt`), và xây dựng module quản lý cấu hình tập trung (`config.py`) sử dụng `pydantic-settings` để đọc các biến môi trường an toàn từ file `.env`.

---

## 2. Danh Sách Task Cần Code

### Task 0.1: Khởi tạo Virtual Environment & File `requirements.txt`
* **File cần tạo/sửa**: `requirements.txt` tại thư mục gốc.
* **Nội dung cần có**:
  - Web & API: `fastapi`, `uvicorn[standard]`, `python-multipart`
  - Data & ML: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `pyarrow`
  - Text & Parsing: `pymupdf`, `docx2txt`, `rapidfuzz`, `sentence-transformers`
  - RAG & Vector Store: `faiss-cpu`, `rank-bm25`
  - AI & Agent: `langchain`, `langchain-community`, `langchain-google-genai`, `langchain-openai`, `langgraph`, `pydantic`, `pydantic-settings`
  - Testing & Utility: `pytest`, `python-dotenv`, `httpx`
* **Các bước thực hiện**:
  1. Tạo môi trường ảo: `python -m venv .venv`
  2. Kích hoạt môi trường: `.venv\Scripts\Activate.ps1`
  3. Cài đặt thư viện: `pip install -r requirements.txt`
* **Definition of Done (DoD)**:
  ```powershell
  python -c "import fastapi, sklearn, xgboost, langgraph, fitz; print('All core libraries installed successfully!')"
  ```

---

### Task 0.2: Xây dựng Module Quản lý Cấu hình (`source/infrastructure/config.py`)
* **File cần code**: `source/infrastructure/config.py`
* **Mục tiêu**: Đọc các khóa bí mật (API Keys), tham số mô hình (weights, thresholds), và đường dẫn dữ liệu từ `.env` với validation kiểu dữ liệu nghiêm ngặt.
* **Input / Output Contract**:
  - Input: Biến môi trường hệ thống hoặc file `.env`.
  - Output: Đối tượng singleton `settings = Settings()`.
* **Các trường thông tin bắt buộc trong `Settings`**:
  ```python
  class Settings(BaseSettings):
      # App Settings
      APP_NAME: str = "AI Career Copilot"
      APP_ENV: str = "development"
      DEBUG: bool = True
      
      # Model & API Keys
      GEMINI_API_KEY: Optional[str] = None
      OPENAI_API_KEY: Optional[str] = None
      FAST_LLM_MODEL: str = "gemini-1.5-flash"
      REASONING_LLM_MODEL: str = "gemini-1.5-pro"
      
      # Paths
      BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
      DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw_data"
      DATA_CLEAN_DIR: Path = BASE_DIR / "data" / "clean_data"
      
      # Algorithm Parameters
      ROUTER_CONFIDENCE_THRESHOLD: float = 0.85
      CLUSTERING_N_CLUSTERS: int = 12
      WEIGHT_SIMILARITY: float = 0.50
      WEIGHT_AI_SAFETY: float = 0.35
      WEIGHT_SALARY: float = 0.15
      SAFETY_MARGIN_DELTA: float = 10.0
  ```
* **Definition of Done (DoD)**:
  ```powershell
  python -c "from source.infrastructure.config import settings; print(settings.APP_NAME, settings.DATA_RAW_DIR)"
  ```
