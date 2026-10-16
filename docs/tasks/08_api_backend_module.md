# Module 8: API Presentation Layer (FastAPI Controllers)

---

## 1. Mục Tiêu Module
Xây dựng tầng giao tiếp RESTful API:
* **Controller thuần túy**: Nhận HTTP Request, validate schema đầu vào, gọi xuống **Application Service (Module 7)** để xử lý, và trả về HTTP Response.
* Tích hợp middlewares (CORS, Global Error Handling, Structured Logging).

---

## 2. Danh Sách Task Cần Code

### Task 8.1: FastAPI App & Middlewares (`source/api/app.py`)
* **File cần code**: `source/api/app.py`, `source/api/middleware/cors.py`, `error_handler.py`.
* **Nhiệm vụ**: Khởi tạo ứng dụng FastAPI và đăng ký router v1.

---

### Task 8.2: Endpoints V1 (`source/api/v1/routers/`)
* **File cần code**:
  - `source/api/v1/routers/cv_router.py`:
    - `POST /api/v1/cv/upload`: Gọi `CVAnalysisService.execute_initial_extraction`.
    - `POST /api/v1/cv/confirm`: Gọi `HITLService.confirm_and_resume`.
  - `source/api/v1/routers/analysis_router.py`:
    - `GET /api/v1/analysis/{session_id}`: Trả về kết quả phân tích rủi ro & lộ trình.
    - `POST /api/v1/copilot/chat`: Tương tác hỏi đáp sâu về kỹ năng phòng thủ.

---

### Task 8.3: Kiểm Thử Toàn Trình (E2E Integration Test) (`tests/test_e2e.py`)
* **File cần code**: `tests/test_e2e.py`
* **Lệnh chạy test**:
  ```powershell
  pytest tests/test_e2e.py -v
  ```
