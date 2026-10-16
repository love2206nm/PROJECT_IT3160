# Hướng Dẫn Chi Tiết 9 Tệp Dữ Liệu Thô (Raw Datasets Guide)

Tài liệu này phân tích chi tiết từng tệp dữ liệu thô trong thư mục `data/raw_data/`: cấu trúc cột, mục đích sử dụng, các cột cần giữ lại, các cột thừa cần loại bỏ để tối ưu bộ nhớ, và các chiến lược xử lý ngoại lệ.

---

## 1. `ai-career-threat-index.csv`

### 1.1. Thông Số Cơ Bản
* **Kích thước**: 148 KB | **Số dòng**: 300 dòng | **Khóa chính**: `slug` (hoặc `soc_code`)
* **Vai trò**: **Trái tim định lượng rủi ro AI** của toàn bộ hệ thống. Chứa điểm đe dọa (Score 4 - 92), cấp độ rủi ro (Low $\rightarrow$ Very High), phân rã 3 chiều rủi ro, 3 kỹ năng phòng thủ thực chiến (`defense_skills`) và nhận định chuyên gia (`insight`).

### 1.2. Phân Tích Cột: Giữ Lại vs Loại Bỏ

| Tên Cột | Kiểu Dữ Liệu | Giữ / Bỏ | Lý Do Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `slug` | `string` | **GIỮ** | Định danh duy nhất theo URL-friendly slug, dùng làm khóa liên kết với `soc-crosswalk.csv`. |
| `soc_code` | `string` | **GIỮ** | Mã O\*NET SOC 6 ký tự để tra cứu bảng. |
| `title` | `string` | **GIỮ** | Tên chức danh chuẩn hóa hiển thị trên báo cáo. |
| `score` | `float` | **GIỮ** | Điểm đe dọa tổng thể (4.0 đến 92.0). |
| `risk_level` | `string` | **GIỮ** | Nhãn rủi ro phân loại (`Low`, `Moderate`, `High`, `Very High`). |
| `sub_task_automation` | `float` | **GIỮ** | Chiều 1: Mức độ tự động hóa các tác vụ lặp lại hàng ngày (0-100). |
| `sub_tool_maturity` | `float` | **GIỮ** | Chiều 2: Độ chín muồi của các công cụ AI chuyên ngành hiện có (0-100). |
| `sub_agentic_exposure` | `float` | **GIỮ** | Chiều 3: Mức độ phơi nhiễm trước các tác tử tự chủ (Agentic AI) (0-100). |
| `defense_skill_1, 2, 3`| `string` | **GIỮ** | Bộ 3 kỹ năng phòng thủ thực chiến, là kim chỉ nam cho RAG và Copilot Agent sinh lộ trình. |
| `insight` | `string` | **GIỮ** | Nhận định chuyên môn định tính về cơ chế AI thay thế, dùng làm mỏ neo sự thật cho RAG Corpus. |
| `salary_low_usd` | `float` | **GIỮ** | Mức cận dưới dải lương năm (USD), dùng làm **fallback lương** khi mã SOC thiếu lương metro. |
| `salary_high_usd` | `float` | **GIỮ** | Mức cận trên dải lương năm (USD), dùng làm **fallback lương**. |
| `salary_trend` | `string` | **GIỮ** | Xu hướng lương thị trường (`growing`, `stable`, `declining`). |
| `score_q1_2025` đến `score_q3_2026` | `float` | **LOẠI BỎ** | Dữ liệu chuỗi thời gian quá khứ/tương lai, không phục vụ bài toán định tuyến hay phân cụm hiện hành. |
| `tier` | `int` | **LOẠI BỎ** | Phân tầng nội bộ của bộ dữ liệu nguồn, dư thừa khi đã có `risk_level` và `score`. |
| `sub_adoption` | `float` | **LOẠI BỎ** | Trùng lặp ý nghĩa tương quan cao với `sub_tool_maturity` (> 0.88 Pearson correlation). |

---

## 2. `soc-crosswalk.csv`

### 2.1. Thông Số Cơ Bản
* **Kích thước**: 76 KB | **Số dòng**: 867 dòng | **Khóa chính**: `soc_code`
* **Vai trò**: **Cầu nối chuẩn hóa toàn diện**. Ánh xạ toàn bộ 867 mã ngành O\*NET SOC của nền kinh tế sang 300 hồ sơ rủi ro trong `ai-career-threat-index.csv`.

### 2.2. Phân Tích Cột
* **Toàn bộ 7 cột đều được GIỮ LẠI**: `soc_code`, `soc_title`, `nearest_role_slug`, `nearest_role_title`, `match_quality`, `score`, `risk_level`.
* Phân bố độ khớp `match_quality`:
  - `close`: 371 nghề (42.8%)
  - `exact`: 248 nghề (28.6%)
  - `approximate`: 215 nghề (24.8%)
  - `unmatched`: 33 nghề (3.8%)

### 2.3. Đánh Giá Rủi Ro Từng Nghề Cho 33 Nghề `unmatched`
Qua quét phân tích thực tế, 33 nghề chưa khớp thuộc 2 nhóm ngành đặc thù trong thế giới vật lý:

#### Nhóm A: Nhóm Ngành 45-xxxx (Nông - Lâm - Ngư Nghiệp & Khai Thác Gỗ - 14 nghề)
* **Danh sách nghề**:
  1. `45-1011`: First-Line Supervisors of Farming, Fishing, and Forestry Workers
  2. `45-2011`: Agricultural Inspectors
  3. `45-2021`: Animal Breeders
  4. `45-2041`: Graders and Sorters, Agricultural Products
  5. `45-2091`: Agricultural Equipment Operators
  6. `45-2092`: Farmworkers and Laborers, Crop, Nursery, and Greenhouse
  7. `45-2093`: Farmworkers, Farm, Ranch, and Aquacultural Animals
  8. `45-2099`: Agricultural Workers, All Other
  9. `45-3031`: Fishing and Hunting Workers
  10. `45-4011`: Forest and Conservation Workers
  11. `45-4021`: Fallers
  12. `45-4022`: Logging Equipment Operators
  13. `45-4023`: Log Graders and Scalers
  14. `45-4029`: Logging Workers, All Other
* **Logic đánh giá rủi ro AI**:
  - Các công việc này yêu cầu thao tác khéo léo của tay và cơ thể trong môi trường vật lý không định hình (Unstructured Physical Environments). LLM và Agentic AI thuần túy phần mềm hoàn toàn không thể thay thế.
  - **Điểm rủi ro gán định chuẩn**: `threat_score = 14.0`, `risk_level = "Low"`.
  - **Phân rã rủi ro**: `sub_task_automation = 15.0`, `sub_tool_maturity = 12.0`, `sub_agentic_exposure = 8.0`.
  - **3 Defense Skills**: *Precision Agriculture Robotics Maintenance, Smart Sensor Monitoring, Bio-diversity Conservation Principles*.

#### Nhóm B: Nhóm Ngành 55-xxxx (Lực Lượng Quân Sự & Tác Chiến Đặc Biệt - 19 nghề)
* **Danh sách nghề**:
  - Sĩ quan chỉ huy (`55-1011` đến `55-1019`): Air Crew, Aircraft Launch, Armored Assault, Artillery/Missile, Command & Control, Infantry, Special Forces Officers.
  - Giám sát tác chiến (`55-2011` đến `55-2013`): Air Crew Supervisors, Weapons Specialists Supervisors.
  - Chiến sĩ trực tiếp (`55-3011` đến `55-3019`): Tactical Operations Specialists, Special Forces, Infantry.
* **Logic đánh giá rủi ro AI**:
  - Hoạt động tác chiến thực địa đòi hỏi quyết sách sinh tử của con người (Human-in-the-Loop bắt buộc theo luật pháp quốc tế và quy tắc đạo đức quân sự).
  - **Điểm rủi ro gán định chuẩn**: `threat_score = 12.0`, `risk_level = "Low"`.
  - **Phân rã rủi ro**: `sub_task_automation = 10.0`, `sub_tool_maturity = 20.0`, `sub_agentic_exposure = 5.0`.
  - **3 Defense Skills**: *Tactical Real-time Decision Making, Autonomous Systems Command & Oversight, Cyber-Physical Battlefield Resilience*.

---

## 3. `job_titles.csv`

### 3.1. Thông Số Cơ Bản
* **Kích thước**: 4.3 MB | **Số dòng**: 54,269 dòng | **Số chức danh thực tế**: 44,779
* **Vai trò**: Cung cấp từ điển phong phú để thuật toán **RapidFuzz** đối sánh nhanh chuỗi chức danh người dùng gõ trên CV (ví dụ `"Senior React Developer"`, `"Cloud Solution Architect"`) với mã chuẩn O\*NET-SOC.

### 3.2. Phân Tích Cột
* **Giữ lại**: `O*NET-SOC Code`, `Job Title`.
* **Loại bỏ**:
  - `Short Title`: 99% giá trị là `NaN`.
  - `Source(s)`: Mã số nguồn khảo sát nội bộ O\*NET, không có giá trị phân tích nghiệp vụ.

---

## 4. `essential_skills.csv`, `knowledge.csv`, `abilities.csv` (Bộ 3 Thước Đo Năng Lực)

### 4.1. Thông Số Cơ Bản
* `essential_skills.csv`: 2.2 MB | 18,200 dòng | **10 kỹ năng cơ bản** $\times$ 910 nghề $\times$ 2 thang đo (`IM`, `LV`).
* `knowledge.csv`: 7.6 MB | 60,060 dòng | **33 miền tri thức** $\times$ 910 nghề $\times$ 2 thang đo (`IM`, `LV`).
* `abilities.csv`: 12.1 MB | 94,640 dòng | **52 năng lực thể chất/nhận thức** $\times$ 910 nghề $\times$ 2 thang đo (`IM`, `LV`).

### 4.2. Hai Thang Đo Cốt Lõi: IM và LV
* **Thang `IM` (Importance)**: Thang đo từ 1.0 đến 5.0 đo lường tầm quan trọng của năng lực trong công việc.
* **Thang `LV` (Level)**: Thang đo từ 0.0 đến 7.0 đo lường độ khó/đòi hỏi chuyên môn của năng lực.

### 4.3. Các Cột Cần Loại Bỏ (Dữ Liệu Thống Kê Điều Tra Thừa)
Trong cả 3 tệp, các cột sau là **nhiễu khảo sát xã hội học** cần loại bỏ ngay khi load:
- `N` (Kích thước mẫu điều tra)
- `Standard Error` (Sai số chuẩn)
- `Lower CI Bound` & `Upper CI Bound` (Khoảng tin cậy 95%)
- `Recommend Suppress` & `Not Relevant`
- `Date` & `Domain Source` (Ngày khảo sát và nguồn)

---

## 5. `software_skills.csv`

### 5.1. Thông Số Cơ Bản
* **Kích thước**: 3.4 MB | **Số dòng**: 31,821 dòng | **Số công cụ độc nhất**: 8,753 phần mềm
* **Vai trò**: Cung cấp danh mục công nghệ thực tế để bóc tách và đối chiếu kỹ năng kỹ thuật của ứng viên.

### 5.2. Phân Tích Cột & Chiến Lược Khai Thác Đa Tầng (Multi-Tier Architecture)
* **Giữ lại**:
  - `O*NET-SOC Code`: Mã định danh nghề nghiệp.
  - `Workplace Example`: Tên công cụ / ngôn ngữ / nền tảng cụ thể (ví dụ: `Python`, `Docker`, `PostgreSQL`, `AWS`, `Kubernetes`).
  - `Element Name`: Tên 134 nhóm danh mục phần mềm chuẩn O*NET (ví dụ: `Development environment software`, `Data base user interface and query software`). **Cực kỳ quan trọng để gom cụm không gây nhiễu!**
  - `Hot Technology`: Cờ nhị phân `'Y'` / `'N'` đánh dấu công nghệ đang bùng nổ nhu cầu.
  - `In Demand`: Cờ nhị phân `'Y'` / `'N'` đánh dấu công nghệ được nhà tuyển dụng săn đón.
* **Loại bỏ**:
  - `Element ID` (mã số phân cấp dạng `2.E.5.b`, dư thừa khi đã có `Element Name`).

### 5.3. Chiến Lược Tận Dụng Triệt Để 8,753 Công Cụ Trên 4 Phân Tầng
Thay vì chỉ trích xuất Top 20 công cụ đơn lẻ (vốn bị loãng vì toàn Word, Excel, Outlook và lãng phí 99.7% dữ liệu), hệ thống khai thác toàn diện kho công cụ này tại 4 vị trí kiến trúc:
1. **Tầng Ingestion & Bóc Tách CV (Module 3)**: Sử dụng toàn bộ 8,753 tên công cụ làm **Từ điển Thực thể Chuẩn Hóa (Entity Normalization / NER Dictionary)** để nhận diện chính xác các từ viết tắt từ CV ứng viên (`k8s` $\rightarrow$ `Kubernetes`, `py` $\rightarrow$ `Python`, `postgres` $\rightarrow$ `PostgreSQL`), đồng thời làm từ điển gợi ý từ khóa thông minh (Autocomplete) ở bước Human-in-the-Loop.
2. **Tầng So Sánh Khoảng Cách Năng Lực (Skill Gap - Module 6 & 7)**: Áp dụng đại số tập hợp thuần túy ($A \cap B$ và $B \setminus A$) trên toàn bộ kho công cụ để bóc tách chính xác:
   - `matched_tools`: Các công cụ ứng viên đã thành thạo kế thừa sang nghề đích.
   - `missing_critical_tools`: Các công cụ nghề đích bắt buộc phải có mà ứng viên còn thiếu, ưu tiên sắp xếp công cụ `Hot Technology = 'Y'` lên đầu.
3. **Tầng RAG Evidence Grounding & Copilot Agent (Module 5 & 6)**: Đưa toàn bộ danh mục công cụ Hot Tech và In Demand của nghề đích vào **RAG Corpus** và **System Prompt** để sinh bài tập thực hành (`practical_projects`) và hành động 30-60-90 ngày chính xác.
4. **Tầng Học Máy Phân Cụm (Feature Matrix 120D - Module 4)**: Thay vì dùng Dense Embedding (dễ bị nhiễu và pha loãng thông tin), ta gom 134 danh mục `Element Name` thành **20 Cụm Công Nghệ Nền Tảng (Core Software Domains)** để tính tỷ lệ phân bố công nghệ của từng nghề, đảm bảo 100% không bị nhiễu và giữ trọn vẹn khả năng giải thích (Explainability).

---

## 6. `task_ratings.csv`

### 6.1. Thông Số Cơ Bản
* **Kích thước**: 37.2 MB | **Số dòng**: 165,780 dòng | **Số tác vụ độc nhất**: 18,420 tác vụ
* **Vai trò**: Cung cấp mô tả chi tiết từng đầu việc hàng ngày của từng nghề. Dùng cho:
  1. Phân tích tác vụ có nguy cơ tự động hóa cao.
  2. Bơm vào kho dẫn chứng **RAG Corpus** để Copilot Agent trích dẫn chứng cứ xác thực khi tư vấn cho người dùng.

### 6.2. Phân Tích Cột
* **Giữ lại**: `O*NET-SOC Code`, `Task ID`, `Task`, `Scale ID`, `Data Value`.
* **Lọc thang đo**:
  - Giữ lại `Scale ID == 'IM'` (Tầm quan trọng) và `Scale ID == 'RT'` (Độ liên quan Relevance).
  - Loại bỏ các dòng `Scale ID == 'FT'` (Tần suất phân theo 7 nhóm loại, gây phân mảnh và phình to dữ liệu).
* **Loại bỏ các cột thống kê**: `N`, `Standard Error`, `Lower CI Bound`, `Upper CI Bound`, `Recommend Suppress`, `Date`, `Domain Source`, `Category`.

---

## 7. `voiced-real-pay-by-metro-2025.csv`

### 7.1. Thông Số Cơ Bản
* **Kích thước**: 7.9 MB | **Số dòng**: 47,380 dòng | **Số mã SOC đô thị**: 368
* **Vai trò**: Cung cấp mức thu nhập thực tế năm 2025 đã điều chỉnh theo sức mua địa phương (Regional Price Parities - RPP), phục vụ cho trọng số lương (0.15) trong hàm thỏa dụng `Utility` của mô hình phân cụm chuyển đổi an toàn.

### 7.2. Phân Tích Cột & Chiến Lược Tính Toán
* **Cột giữ lại**: `soc_code`, `real_p50` (Lương trung vị thực tế tính bằng USD/năm).
* **Cột loại bỏ**: `area_code`, `cbsa`, `area_name`, `employment`, `p25`, `p50`, `p75`, `rpp_all_items`, `rpp_goods`, `rpp_housing`, `real_p25`, `real_p75`, `band_ratio_p75_p25`, `base_pay_only_flag`, `oews_period`, `rpp_year`.
* **Chiến lược tổng hợp quốc gia**:
  $$\text{National Median Salary}(\text{SOC}) = \text{median}(\text{real\_p50})_{\text{across all metro areas}}$$
* **Chiến lược bù khuyết (Fallback Imputation)**:
  - 368 mã SOC có trong bảng này: lấy $\text{median}(\text{real\_p50})$.
  - 507 mã SOC không có trong bảng metro: lấy $\frac{\text{salary\_low\_usd} + \text{salary\_high\_usd}}{2}$ từ `ai-career-threat-index.csv` (được map qua `soc-crosswalk.csv`).
  $\Rightarrow$ **Đạt độ phủ 100% (867 / 867 mã SOC đều có mức lương chuẩn)**.
