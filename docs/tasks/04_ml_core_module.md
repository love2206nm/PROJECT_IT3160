# Module 4: Machine Learning Core (Phân Cụm & Dự Báo Vị Trí Lai)

---

## 1. Mục Tiêu Module
Triển khai hai giao diện Domain: `IClusteringEngine` và `IRiskPredictor`:
1. **Agglomerative Clustering Engine & Transition Utility**: Dùng ma trận 76D từ `data_pipeline` với **Spherical Ward Linkage** ($K=23$ nhóm tương ứng 23 Major Groups O*NET) để phân cụm cân bằng, triệt tiêu lỗi sụp đổ cụm của Average Linkage, tìm Top nghề an toàn lân cận và tính toán `SkillGap`.
2. **Regularized XGBoost Risk Predictor & SHAP Explainer**: Dự báo điểm số rủi ro (0-100) bằng **Regularized XGBRegressor** (`max_depth=3, subsample=0.8, reg_alpha=1.0, reg_lambda=2.0`), kết hợp **Deterministic Boundary Mapping** cho cấp độ rủi ro và `shap.TreeExplainer` để bóc tách 3 kỹ năng phòng thủ và 3 kỹ năng rủi ro cho các vị trí lai.
3. **Model Persistence & CLI Huấn Luyện**: Lưu trữ trọng số mô hình vào `models/` và cung cấp CLI `train_models_cli.py` với logging chuẩn hóa.
4. **User Application Service**: Triển khai `UserCareerService` trong `source/services/user/career_service.py` nạp model đã lưu để phục vụ người dùng.

---

## 2. Danh Sách Task Cần Code

### Task 4.1: Agglomerative Clustering & Utility (`source/model/ML_model/clustering_engine.py`)
* **File cần code**: `source/model/ML_model/clustering_engine.py`
* **Triển khai Interface**: `IClusteringEngine` từ `domain/interfaces/ml_models.py`.
* **Dữ liệu**: Đọc ma trận đặc trưng từ `data/clean_data/final/soc_feature_matrix.parquet` và `soc_metadata.parquet`.
* **Nhiệm vụ**:
  - Chuẩn hóa $L_2$ các vector đặc trưng và phân cụm $K=23$ bằng `AgglomerativeClustering(metric='euclidean', linkage='ward')` (Spherical Ward Linkage, tối thiểu hóa phương sai góc, NMI = 0.466).
  - Triển khai hàm tính độ thỏa dụng chuyển đổi nghề nghiệp:
    $$\text{Utility}(T) = 0.50 \cdot \text{CosineSim}(\vec{S}_C, \vec{S}_T) + 0.35 \cdot \left(1 - \frac{\text{Threat}(T)}{100}\right) + 0.15 \cdot \text{SalaryAdvantage}(T)$$
  - Tính toán chi tiết `SkillGap` (`missing_skills`, `transferable_skills`, `gap_severity`).
  - Hỗ trợ `save_model(path)` và `load_model(path)` qua `joblib`.
  - Trả về danh sách `List[SafeTransitionRole]`.

---

### Task 4.2: Regularized XGBoost Risk Predictor (`source/model/ML_model/risk_predictor.py`)
* **File cần code**: `source/model/ML_model/risk_predictor.py`
* **Triển khai Interface**: `IRiskPredictor` từ `domain/interfaces/ml_models.py`.
* **Nhiệm vụ**:
  - `XGBRegressor` với siêu tham số kiểm soát overfitting: `max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0`.
  - Ánh xạ điểm số liên tục sang `RiskLevel` bằng Deterministic Boundary Mapping (Low < 30, Moderate 30-59, High 60-79, Very High >= 80).
  - Sử dụng `shap.TreeExplainer` để bóc tách 3 kỹ năng làm tăng rủi ro và 3 kỹ năng phòng thủ / giảm rủi ro.
  - Hỗ trợ vector hóa hồ sơ ứng viên `CandidateProfile` thành vector 76 chiều dựa trên từ điển kỹ năng O*NET.
  - Hỗ trợ `save_model(path)` và `load_model(path)` qua `joblib`.
  - Đóng gói đầu ra thành Domain Entity `RiskProfile(is_deterministic=False, ...)`.

---

### Task 4.3: CLI Huấn Luyện & Lưu Trữ Mô Hình (`source/services/cli/train_models_cli.py`)
* **File cần code**: `source/services/cli/train_models_cli.py`
* **Nhiệm vụ**:
  - CLI hỗ trợ cờ lệnh `--model [all|clustering|risk]`, `--clusters 23`, `--output-dir models`.
  - Huấn luyện, đánh giá 5-fold cross-validation, ghi log có cấu trúc qua `get_logger(__name__)` (Zero print statements).
  - Lưu trữ `clustering_model.joblib` và `xgb_risk_model.joblib` vào `models/`.

---

### Task 4.4: User Career Transition Service (`source/services/user/career_service.py`)
* **File cần code**: `source/services/user/career_service.py`
* **Nhiệm vụ**:
  - Cung cấp `UserCareerService` nạp sẵn model từ đĩa hoặc lazy load.
  - Cung cấp các use cases: `get_safe_career_pivots`, `predict_hybrid_role_risk`, `analyze_candidate_transition`.

---

### Task 4.5: Unit & Integration Tests (`tests/test_ml.py` & `tests/test_career_service.py`)
* **File cần code**: `tests/test_ml.py`, `tests/test_career_service.py`
* **Lệnh chạy test**:
  ```powershell
  pytest tests/test_ml.py tests/test_career_service.py -v
  ```

