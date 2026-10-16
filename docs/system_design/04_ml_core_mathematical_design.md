# Cơ Sở Toán Học & Thiết Kế Giải Thuật Machine Learning Core (Task 04)

> **Tài liệu Kỹ thuật Chuyên sâu**  
> **Tác giả:** Senior Data Scientist & ML Architect  
> **Dự án:** AI Career Copilot & Job Advisor (IT3160)  
> **Dữ liệu thực nghiệm:** O*NET 29.0 Database ($N = 867$ nghề nghiệp, $D = 76$ chiều năng lực Gold Store)

---

## 1. Tổng Quan Không Gian Đặc Trưng & Thách Thức Toán Học

Hệ thống đặc trưng đầu vào phục vụ phân tích nghề nghiệp được rút trích từ ma trận `data/clean_data/final/soc_feature_matrix.parquet`:
- **Số lượng mẫu (Sample Size)**: $N = 867$ nghề nghiệp chuẩn hóa theo hệ thống SOC (Standard Occupational Classification).
- **Số chiều không gian (Dimensionality)**: $D = 76$ đặc trưng liên tục, đại diện cho 16 thang đo năng lực O*NET sau khi đã thực hiện Group-wise Pruning loại bỏ đa cộng tuyến ($|r| \ge 0.85$).
- **Không gian giá trị**: $\mathbf{X} \in [0.0, 1.0]^{N \times D}$. Mọi phần tử đều không âm, tập trung hoàn toàn trong **góc phần tư siêu không gian thứ nhất (First Hyper-octant)**.
- **Tỷ lệ mẫu trên số chiều**:
  $$\frac{N}{D} = \frac{867}{76} \approx 11.41$$
  Đây là một tỷ lệ tương đối nhỏ trong học máy thống kê. Nếu không kiểm soát chính quy hóa (regularization) chặt chẽ, các mô hình học sâu hoặc mô hình cây phức tạp sẽ lập tức rơi vào hiện tượng **Quá khớp (Overfitting)** và **Lời nguyền số chiều (Curse of Dimensionality)**.

---

## 2. Giải Thuật 1: Phân Cụm Không Gian Nghề Nghiệp (`IClusteringEngine`)

### 2.1. Phân Tích Sự Thất Bại Toán Học Của `Cosine Distance + Average Linkage`

Tài liệu thiết kế ban đầu đề xuất: `AgglomerativeClustering(metric='cosine', linkage='average', n_clusters=12)`.

Tuy nhiên, khi kiểm định thực nghiệm trên 867 nghề nghiệp thực tế, cấu hình này dẫn đến **Hiện tượng sụp đổ kích thước cụm (Cluster Size Collapse / Chaining Effect)** cực kỳ nghiêm trọng:

```
Phân bố kích thước cụm tại K=12 (Cosine + Average Linkage):
Cụm lớn nhất: 743 nghề (chiếm 85.7% toàn bộ dữ liệu!)
Các cụm còn lại: [69, 26, 12, 4, 4, 3, 2, 1, 1, 1, 1]
Normalized Mutual Information (NMI so với 23 nhóm ngành thực tế O*NET): 0.124
```

#### Nguyên nhân giải tích & hình học:
1. **Đặc tính First Hyper-octant**: Vì mọi vector năng lực $\vec{x}_i \ge 0$, góc giữa hai vector bất kỳ luôn thỏa mãn $\theta \in [0^\circ, 90^\circ]$, do đó độ tương đồng Cosine luôn dương:
   $$\text{CosineSim}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} \in (0, 1] \implies d_{\cos}(\vec{u}, \vec{v}) = 1 - \text{CosineSim} \in [0, 1)$$
2. **Hiệu ứng xích dính (Chaining / Giant Component)**: Trong `Average Linkage`, khoảng cách giữa hai cụm $A$ và $B$ được định nghĩa:
   $$D_{\text{avg}}(A, B) = \frac{1}{|A||B|} \sum_{\vec{u} \in A} \sum_{\vec{v} \in B} d_{\cos}(\vec{u}, \vec{v})$$
   Vì các vector nghề nghiệp trong không gian 76 chiều có mật độ liên tục và không có ranh giới cô lập tuyệt đối, cụm nào có số lượng phần tử lớn hơn sẽ có xu hướng kéo giá trị trung bình khoảng cách về phía tâm khối của nó, tạo thành một **"Siêu cụm khổng lồ" (Giant Component)** hút gần như toàn bộ các điểm lân cận vào, biến các cụm còn lại thành các điểm ngoại lai cục bộ (1-2 điểm).
3. **Hệ quả nghiệp vụ**: Một kỹ sư phần mềm sẽ bị xếp chung cụm với cả công nhân mỏ, thợ may và y tá, phá vỡ hoàn toàn ý nghĩa nghiệp vụ của tính năng tìm kiếm vị trí an toàn trong cùng ngành nghề!

---

### 2.2. Giải Pháp Tối Ưu: **Spherical Ward Linkage** (Ward trên Vector Chuẩn Hóa $L_2$)

#### Cơ sở toán học giải tích:
Xét hai vector đặc trưng bất kỳ $\vec{x}_i, \vec{x}_j \in \mathbb{R}^D$. Ta chuẩn hóa $L_2$ các vector này về mặt cầu đơn vị (Unit Hypersphere $\mathbb{S}^{D-1}$):
$$\vec{u}_i = \frac{\vec{x}_i}{\|\vec{x}_i\|_2}, \quad \vec{u}_j = \frac{\vec{x}_j}{\|\vec{x}_j\|_2} \implies \|\vec{u}_i\|_2 = \|\vec{u}_j\|_2 = 1$$

Khai triển bình phương khoảng cách Euclid giữa hai vector đơn vị:
$$\|\vec{u}_i - \vec{u}_j\|_2^2 = \|\vec{u}_i\|_2^2 + \|\vec{u}_j\|_2^2 - 2 (\vec{u}_i \cdot \vec{u}_j) = 1 + 1 - 2 \cos(\theta_{ij}) = 2 \left(1 - \text{CosineSim}(\vec{x}_i, \vec{x}_j)\right) = 2 \cdot d_{\cos}(\vec{x}_i, \vec{x}_j)$$

**Hệ quả trực tiếp**:
> Bình phương khoảng cách Euclid trên mặt cầu đơn vị tỉ lệ thuận tuyệt đối $1:1$ với Khoảng cách Cosine!

Thuật toán liên kết **Ward's Minimum Variance Linkage** tìm cách sáp nhập hai cụm $A$ và $B$ sao cho độ tăng tổng phương sai trong cụm ($\Delta \text{ESS}$) là nhỏ nhất:
$$\Delta \text{ESS}(A, B) = \frac{|A||B|}{|A| + |B|} \|\vec{\mu}_A - \vec{\mu}_B\|_2^2$$
Khi áp dụng trên $\vec{u} \in \mathbb{S}^{D-1}$, việc tối thiểu hóa phương sai Euclid chính là **Tối thiểu hóa phương sai góc (Angular Variance Minimization)** trong không gian năng lực, đồng thời hàm mục tiêu có thành phần trọng số kích thước $\frac{|A||B|}{|A| + |B|}$ đóng vai trò là **lực cản chính quy hóa hình học**, phạt nặng việc sáp nhập các cụm đã quá lớn, triệt tiêu hoàn toàn hiện tượng "Siêu cụm khổng lồ"!

#### Bằng chứng thực nghiệm so sánh:

| Tiêu chí | Cosine + Average Linkage ($K=12$) | Spherical Ward Linkage ($K=12$) | Spherical Ward Linkage ($K=23$) |
| :--- | :---: | :---: | :---: |
| **Silhouette Score** | 0.237 | **0.249** | 0.222 |
| **NMI (so với 23 nhóm O\*NET)** | 0.124 | **0.419** (+238%) | **0.466** (+275%) |
| **Kích thước cụm lớn nhất** | 743 (85.7% sụp đổ) | **179 (20.6% cân bằng)** | **74 (8.5% lý tưởng)** |
| **Kích thước cụm nhỏ nhất** | 1 (ngoại lai rác) | **10** | **4** |
| **Số cụm < 5 phần tử** | 8 cụm | **0 cụm** | 1 cụm |

**Quyết định lựa chọn**:
Sử dụng `AgglomerativeClustering(metric='euclidean', linkage='ward', n_clusters=23)` trên ma trận chuẩn hóa $L_2$. Giá trị $K=23$ khớp chính xác với 23 Major Groups của hệ thống SOC Hoa Kỳ, bảo toàn cấu trúc phân loại lao động thực tế.

---

### 2.3. Hàm Thỏa Dụng Chuyển Đổi Nghề Nghiệp (Transition Utility Engine)

Sau khi xác định được cụm nghề nghiệp mục tiêu, hệ thống xếp hạng các lựa chọn chuyển đổi an toàn thông qua hàm thỏa dụng đa mục tiêu:

$$\text{Utility}(T) = w_1 \cdot \text{CosineSim}(\vec{S}_C, \vec{S}_T) + w_2 \cdot \left(1 - \frac{\text{Threat}(T)}{100}\right) + w_3 \cdot \text{SalaryAdvantage}(T)$$

Với các trọng số thỏa hiệp:
- $w_1 = 0.50$: Trọng số tương đồng kỹ năng (khả thi chuyển đổi).
- $w_2 = 0.35$: Trọng số mức độ an toàn trước tự động hóa AI.
- $w_3 = 0.15$: Trọng số lợi thế thu nhập.
- Tổng trọng số: $w_1 + w_2 + w_3 = 1.0$.

Trong đó:
1. $\text{CosineSim}(\vec{S}_C, \vec{S}_T) = \frac{\vec{S}_C \cdot \vec{S}_T}{\|\vec{S}_C\| \|\vec{S}_T\|} \in [0, 1]$.
2. $\text{SalaryAdvantage}(T) = \max\left(0, \min\left(1, \frac{\text{Salary}(T) - \text{Salary}(C)}{\text{Salary}(C)}\right)\right)$: Chuẩn hóa tỷ lệ tăng lương vào đoạn $[0, 1]$.
3. **Bộ lọc an toàn (Safety Constraint)**:
   $$\text{Threat}(T) \le \text{Threat}(C) - \delta \quad (\delta = 10.0)$$
   Đảm bảo nghề đích phải có điểm rủi ro AI thấp hơn đáng kể so với nghề hiện tại của ứng viên.

---

## 3. Giải Thuật 2: Dự Báo Rủi Ro Tự Động Hóa AI (`IRiskPredictor`)

### 3.1. Phân Tích Hiện Tượng Mất Cân Bằng Nhãn & Lựa Chọn Mô Hình Hồi Quy

Biến mục tiêu đo lường mức độ đe dọa của AI (`threat_score`) là một biến liên tục có phân phối trên đoạn $[4.0, 92.0]$, với các thống kê:
- Trung bình ($\mu$): $23.06$
- Độ lệch chuẩn ($\sigma$): $18.00$
- Trung vị: $18.00$ (Phân phối lệch phải rõ rệt)

Khi phân loại thành 4 mức độ rủi ro (`risk_level`):
- `Low`: 692 mẫu (79.8%)
- `Moderate`: 91 mẫu (10.5%)
- `High`: 72 mẫu (8.3%)
- `Very High`: **12 mẫu (1.4%)**

#### Vì sao không sử dụng mô hình Phân loại đa lớp (`XGBClassifier`)?
Trong tập dữ liệu $N = 867$, lớp `Very High` chỉ có 12 mẫu. Nếu thực hiện 5-Fold Cross Validation, mỗi tập kiểm thử fold chỉ có trung bình $2.4$ mẫu. Mô hình phân loại đa lớp gặp phải các nhược điểm toán học chí mạng:
1. **Thiểu số suy biến (Minority Class Collapse)**: Trọng số của 12 mẫu bị lấn át hoàn toàn bởi 692 mẫu lớp Low, dẫn đến xác suất dự đoán cho lớp Very High tiệm cận 0.
2. **Xung đột quyết định (Decision Boundary Inconsistency)**: Nếu huấn luyện song song `XGBRegressor` và `XGBClassifier`, sẽ xảy ra nghịch lý: mô hình hồi quy dự đoán 82 điểm (thuộc Very High), nhưng mô hình phân loại lại gán nhãn High do ranh giới quyết định bị lệch.

#### Giải pháp tối ưu: **Hồi Quy Liên Tục + Ánh Xạ Ngưỡng Xác Định (Deterministic Boundary Mapping)**
Hệ thống sử dụng duy nhất một mô hình hồi quy liên tục để ước lượng hàm mật độ rủi ro, sau đó phân lớp dựa trên các ranh giới xác định đã được hiệu chuẩn:
$$\text{RiskLevel}(s) = \begin{cases}
\text{Low} & \text{khi } s < 30.0 \\
\text{Moderate} & \text{khi } 30.0 \le s < 60.0 \\
\text{High} & \text{khi } 60.0 \le s < 80.0 \\
\text{Very High} & \text{khi } s \ge 80.0
\end{cases}$$

---

### 3.2. Kiểm Soát Quá Khớp (Overfitting Control) Bằng Regularized XGBoost

Kết quả thực nghiệm 5-Fold Cross-Validation:

| Cấu hình | Test MAE | Test RMSE | Test $R^2$ | Train $R^2$ | Khoảng cách Quá Khớp ($\text{Train} - \text{Test}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Mặc định** (`max_depth=6`) | 9.86 | 13.73 | 0.404 | 0.926 | **0.522 (Overfitting nghiêm trọng)** |
| **Ridge Regression** ($\alpha=1.0$) | 9.79 | 13.23 | 0.448 | 0.533 | 0.085 (Underfitting phi tuyến) |
| **Regularized XGBoost** (`max_depth=3`) | **9.38** | **12.79** | **0.485** | **0.749** | **0.264 (Cân bằng tối ưu)** |

#### Các kỹ thuật chính quy hóa giải tích được áp dụng:
1. **Khống chế độ sâu cây (`max_depth = 3`)**: Hạn chế số nút lá tối đa của mỗi cây là $2^3 = 8$, chỉ cho phép mô hình học các tương tác tối đa bậc 3 giữa các nhóm năng lực, ngăn chặn triệt để việc chia nhánh cô lập các mẫu đơn lẻ.
2. **Tốc độ học co hẹp (Shrinkage Rate `learning_rate = 0.05`)**: Làm chậm quá trình hội tụ gradient, giúp mỗi cây mới chỉ đóng góp một bước nhỏ vào phần dư dự báo:
   $$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot f_t(\vec{x}) \quad (\eta = 0.05)$$
3. **Chính quy hóa $L_1$ và $L_2$ trên cấu trúc lá (`reg_alpha = 1.0`, `reg_lambda = 2.0`)**:
   Hàm mục tiêu tối ưu của XGBoost tại bước thứ $t$:
   $$\mathcal{L}^{(t)} = \sum_{i=1}^N l\left(y_i, \hat{y}_i^{(t-1)} + f_t(\vec{x}_i)\right) + \gamma T + \alpha \sum_{j=1}^T |w_j| + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
   - $\alpha \sum |w_j|$ ép các trọng số lá không quan trọng về đúng 0 (Feature Selection).
   - $\lambda \sum w_j^2$ co hẹp độ lớn trọng số dự báo, triệt tiêu phương sai dự báo lớn khi gặp vector ngoại lai.
4. **Lấy mẫu con kép (Dual Subsampling)**:
   - `subsample = 0.8`: Mỗi cây chỉ sử dụng ngẫu nhiên 80% số dòng dữ liệu.
   - `colsample_bytree = 0.8`: Mỗi cây chỉ xem xét 80% số cột đặc trưng (giảm tương quan giữa các cây, tương tự Random Forest).

---

### 3.3. Bóc Tách Khả Năng Giải Thích Bằng SHAP (Shapley Additive exPlanations)

Để đáp ứng yêu cầu bóc tách **3 kỹ năng làm tăng rủi ro** và **3 kỹ năng phòng thủ / giảm rủi ro**, hệ thống tích hợp lý thuyết trò chơi hợp tác của Lloyd Shapley thông qua thuật toán `TreeSHAP` (Lundberg & Lee, 2020).

Giá trị Shapley của đặc trưng $j$ cho một hồ sơ ứng viên $\vec{x}$ được định nghĩa:
$$\phi_j(\vec{x}) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{j\}) - f_x(S) \right]$$

Mô hình thỏa mãn tính chất cộng tính địa phương (Local Accuracy):
$$f(\vec{x}) = \mathbb{E}[f(\mathbf{X})] + \sum_{j=1}^{76} \phi_j(\vec{x})$$

Trong đó:
- $\mathbb{E}[f(\mathbf{X})] \approx 23.06$ là điểm số đe dọa trung bình của toàn thị trường lao động.
- $\phi_j(\vec{x}) > 0$: Kỹ năng $j$ đang **làm tăng nguy cơ bị AI thay thế** so với mức trung bình thị trường.
- $\phi_j(\vec{x}) < 0$: Kỹ năng $j$ là **kỹ năng phòng thủ vững chắc**, kéo điểm rủi ro xuống thấp hơn.

Thuật toán trích xuất:
1. Sắp xếp mảng $\phi_j$ theo thứ tự giảm dần:
   $$\text{VulnerableSkills} = \arg\max_{j,\, \text{top 3}} \phi_j(\vec{x})$$
2. Sắp xếp mảng $\phi_j$ theo thứ tự tăng dần:
   $$\text{DefensiveSkills} = \arg\min_{j,\, \text{top 3}} \phi_j(\vec{x})$$

---

## 4. Tổng Kết Kiến Trúc Toán Học Task 04

```
Input: SOC Gold Matrix (N=867, D=76) & Metadata
│
├── [CLUSTERING ENGINE]
│   ├── L2 Normalization: u = x / ||x||_2
│   ├── Spherical Ward Linkage: min Angular Variance (K=23)
│   ├── Cluster Balancing: Max=74, Min=4, NMI=0.466
│   └── Transition Utility: 0.50*Cosine + 0.35*(1-Threat) + 0.15*SalaryAdvantage
│
└── [RISK PREDICTOR]
    ├── Candidate Profile Vectorization: Mapping 76D capability space
    ├── Regularized XGBRegressor: max_depth=3, subsample=0.8, reg_alpha=1.0, reg_lambda=2.0
    ├── Deterministic Boundary Mapping: [Low <30, Mod 30-59, High 60-79, Severe >=80]
    └── TreeSHAP Explainer: Top 3 Defensive Skills (min phi) & Top 3 Vulnerabilities (max phi)
```
