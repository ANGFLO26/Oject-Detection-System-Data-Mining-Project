# SLIDE THUYẾT TRÌNH
## Hệ thống nhận diện động vật sử dụng YOLO và ứng dụng web

**Sinh viên:** Phan Văn Tài - MSSV: 2202081  
**Giảng viên hướng dẫn:** Tiến sĩ Trần Ngọc Anh

---

## SLIDE 1: TRANG BÌA

**TRƯỜNG ĐẠI HỌC TÂN TẠO**  
**KHOA CÔNG NGHỆ THÔNG TIN**

---

# Hệ thống nhận diện động vật
## Sử dụng YOLO và ứng dụng web

**Sinh viên thực hiện:** Phan Văn Tài  
**Mã số sinh viên:** 2202081  
**Giảng viên hướng dẫn:** Tiến sĩ Trần Ngọc Anh

---

## SLIDE 2: NỘI DUNG TRÌNH BÀY

1. **Giới thiệu đề tài**
2. **Vấn đề và mục tiêu**
3. **Cơ sở lý thuyết (YOLO)**
4. **Xử lý dữ liệu** ⭐ (Tập trung)
5. **Huấn luyện mô hình**
6. **Kết quả**
7. **Demo ứng dụng web** ⭐ (Tập trung)
8. **Kết luận**

---

## SLIDE 3: GIỚI THIỆU ĐỀ TÀI

### Bối cảnh

- Phát hiện đối tượng là bài toán quan trọng trong Computer Vision
- Ứng dụng: Bảo tồn, nghiên cứu, quản lý động vật

### Đề tài

- Xây dựng hệ thống nhận diện **80 lớp động vật**
- Sử dụng **YOLOv8** - mô hình one-stage detection hiện đại
- Triển khai **ứng dụng web** với React + FastAPI

---

## SLIDE 4: VẤN ĐỀ VÀ MỤC TIÊU

### Vấn đề chính

1. **Dataset mất cân bằng nghiêm trọng**
   - Imbalance ratio: **321:1**
   - Butterfly: 2,045 ảnh vs Squid: 28 ảnh

2. **Đa lớp phức tạp**: 80 classes động vật

3. **Yêu cầu độ chính xác cao**: mAP50 ≥ 0.75

### Mục tiêu

- Cân bằng dataset và cải thiện chất lượng
- Đạt mAP50 ≥ 0.75
- Xây dựng ứng dụng web dễ sử dụng

---

## SLIDE 5: CƠ SỞ LÝ THUYẾT - YOLO

### YOLO (You Only Look Once)

- **One-stage detector**: Nhanh, phù hợp real-time
- **YOLOv8 (2023)**: Phiên bản mới nhất
  - Anchor-free architecture
  - C2f module (gradient flow tốt hơn)
  - Decoupled head

### Kiến trúc YOLOv8

```
Backbone (CSPDarknet) → Neck (PANet) → Head (Decoupled)
```

**Ưu điểm:**
- Tốc độ nhanh (~3.7ms/ảnh)
- Độ chính xác cao
- Dễ triển khai

---

## SLIDE 6: XỬ LÝ DỮ LIỆU - TỔNG QUAN

### Dataset

- **Nguồn**: Kaggle - Animals Detection Images Dataset
- **Tổng số ảnh**: 29,071
- **Số classes**: 80
- **Vấn đề**: Mất cân bằng nghiêm trọng

### Pipeline xử lý

```
Raw Dataset → Analysis → Validation → Balancing → Split → YOLO Format
```

---

## SLIDE 7: XỬ LÝ DỮ LIỆU - PHÂN TÍCH

### Thống kê ban đầu

| Metric | Giá trị |
|--------|---------|
| Tổng ảnh | 29,071 |
| Số classes | 80 |
| Max/class | 2,045 (Butterfly) |
| Min/class | 28 (Squid) |
| Trung bình | 363.4 ảnh/class |
| **Imbalance ratio** | **73:1** |

### Phân loại classes

- **Rất ít (< 15)**: 0 classes
- **Ít (15-30)**: 2 classes (Squid, Turtle)
- **Tốt (≥ 30)**: 78 classes

---

## SLIDE 8: XỬ LÝ DỮ LIỆU - VALIDATION

### Image Validation

- ✅ Format: .jpg, .jpeg, .png, .bmp
- ✅ Kích thước: 32x32 ≤ size ≤ 10000x10000
- ✅ Kiểm tra corrupt files

### Bounding Box Validation

- ✅ Swap nếu x_min > x_max
- ✅ Clamp về [0, img_width/height]
- ✅ Loại bỏ nếu:
  - width/height < 5 pixels
  - Area < 0.05% hoặc > 98% ảnh

**Kết quả**: Loại bỏ các samples không hợp lệ

---

## SLIDE 9: XỬ LÝ DỮ LIỆU - CÂN BẰNG

### Chiến lược

1. **Loại bỏ**: Classes < 15 ảnh (0 classes)
2. **Oversampling**: Classes 15-30 → 30 ảnh
   - Squid: 22 → 30 (+8)
   - Turtle: 27 → 30 (+3)
3. **Giữ nguyên**: Classes ≥ 30 ảnh

### Kết quả

| Trước | Sau |
|-------|-----|
| 29,071 samples | 28,184 samples |
| Imbalance 73:1 | Imbalance 73:1 |
| - | Oversampled: 11 samples |

**Cải thiện**: Tất cả classes đều có ≥ 30 samples

---

## SLIDE 10: XỬ LÝ DỮ LIỆU - CHIA DỮ LIỆU

### Stratified Split (80/20)

- **Train**: 22,518 samples (80%)
- **Validation**: 5,666 samples (20%)

### YOLO Format Conversion

Chuyển từ absolute → normalized coordinates:

```
x_center = ((x_min + x_max) / 2) / img_width
y_center = ((y_min + y_max) / 2) / img_height
width = (x_max - x_min) / img_width
height = (y_max - y_min) / img_height
```

### Cấu trúc cuối cùng

```
yolo_dataset_pro/
├── images/train/ (22,518)
├── images/val/ (5,666)
├── labels/train/ (22,518)
├── labels/val/ (5,666)
└── data.yaml
```

---

## SLIDE 11: HUẤN LUYỆN MÔ HÌNH

### Cấu hình

| Tham số | Giá trị |
|---------|---------|
| Model | YOLOv8n (nano) |
| Epochs | 100 |
| Batch size | 32 |
| Image size | 640x640 |
| Optimizer | SGD |
| Learning rate | 0.002 → 0.0001 |
| Hardware | Tesla P100 (16GB) |

### Thời gian

- **Training time**: 8 giờ 21 phút
- **Tốc độ**: ~2.6-2.9 it/s

---

## SLIDE 12: KẾT QUẢ - METRICS

### Metrics tổng hợp

| Metric | Giá trị |
|--------|---------|
| **mAP50** | **0.7565** (75.65%) |
| **mAP50-95** | **0.6322** (63.22%) |
| **Precision** | **0.7140** |
| **Recall** | **0.7469** |
| **F1-Score** | **0.7301** |

### So sánh với baseline

| Dataset | mAP50 | Improvement |
|---------|-------|-------------|
| Imbalanced | 0.6925 | Baseline |
| **Balanced** | **0.7565** | **+9.2%** 🎉 |

**Kết luận**: Cân bằng dữ liệu cải thiện đáng kể!

---

## SLIDE 13: KẾT QUẢ - TOP CLASSES

### Top 5 classes tốt nhất

| Class | mAP50 | Samples |
|-------|-------|---------|
| Woodpecker | 0.991 | 41 |
| Ladybug | 0.975 | 86 |
| Eagle | 0.963 | 179 |
| Zebra | 0.965 | 39 |
| Polar bear | 0.951 | 56 |

### Classes cần cải thiện

| Class | mAP50 | Samples |
|-------|-------|---------|
| Turtle | 0.076 | 6 |
| Squid | 0.172 | 6 |
| Goose | 0.381 | 65 |

**Nhận xét**: Classes có ít samples có hiệu năng thấp

---

## SLIDE 14: ỨNG DỤNG WEB - KIẾN TRÚC

### Kiến trúc hệ thống

```
┌─────────────────┐
│  React Frontend │  Port 3000
│  (Browser)      │
└────────┬────────┘
         │ HTTP/REST API
┌────────▼────────┐
│ FastAPI Backend │  Port 8000
│  - /api/detect  │
│  - /api/batch   │
└────────┬────────┘
         │
┌────────▼────────┐
│  YOLOv8 Model   │
│   (best.pt)     │
└─────────────────┘
```

**Tech Stack:**
- Frontend: React 18.2.0
- Backend: FastAPI 0.104.1
- Model: YOLOv8n

---

## SLIDE 15: DEMO - TÍNH NĂNG CHÍNH

### 1. Upload ảnh

- ✅ Single image (drag & drop)
- ✅ Batch processing (tối đa 20 ảnh)
- ✅ File validation

### 2. Nhận diện

- ✅ Single detection
- ✅ Batch processing
- ✅ Tùy chỉnh thresholds:
  - Confidence: 0.0 - 1.0
  - IoU: 0.0 - 1.0

### 3. Hiển thị kết quả

- ✅ Ảnh với bounding boxes
- ✅ Bảng detections (sortable)
- ✅ Thống kê chi tiết

---

## SLIDE 16: DEMO - GIAO DIỆN

### Các thành phần

1. **Header**: Tên hệ thống, model status
2. **Upload area**: Drag & drop, file picker
3. **Settings panel**: Điều chỉnh thresholds
4. **Image preview**: Ảnh gốc và kết quả (tabs)
5. **Results table**: Bảng detections sortable
6. **Statistics**: Thống kê chi tiết

### Tính năng nổi bật

- ✅ Responsive design
- ✅ Keyboard shortcuts (← →)
- ✅ So sánh thresholds
- ✅ Batch navigation

**[SCREENSHOT GIAO DIỆN WEB APP]**

---

## SLIDE 17: DEMO - VÍ DỤ KẾT QUẢ

### Ví dụ 1: Ảnh có nhiều detections

- **Input**: Ảnh safari với nhiều loài
- **Output**: 
  - Phát hiện 5-8 động vật
  - Confidence > 0.7
  - Bounding boxes chính xác

**[SCREENSHOT KẾT QUẢ 1]**

### Ví dụ 2: Ảnh đơn giản

- **Input**: Ảnh 1-2 động vật rõ ràng
- **Output**:
  - Confidence > 0.8
  - Detection chính xác

**[SCREENSHOT KẾT QUẢ 2]**

---

## SLIDE 18: DEMO - API ENDPOINTS

### REST API

1. **GET /api/model-info**
   - Lấy thông tin model (80 classes)

2. **POST /api/detect**
   - Nhận diện 1 ảnh
   - Request: FormData (file, thresholds)
   - Response: JSON (detections, image_base64)

3. **POST /api/detect-batch**
   - Nhận diện nhiều ảnh
   - Tối đa 20 ảnh

4. **POST /api/compare-thresholds**
   - So sánh kết quả với nhiều thresholds

### Tốc độ

- Single image: ~100-200ms
- Batch (20 images): ~2-4 giây

---

## SLIDE 19: KẾT LUẬN

### Kết quả đạt được

1. ✅ **Pipeline xử lý dữ liệu chuyên nghiệp**
   - Cân bằng dataset: 321:1 → 73:1
   - Validation và cleaning

2. ✅ **Model hiệu quả**
   - mAP50 = 0.7565 (gần mục tiêu 0.78-0.82)
   - Cải thiện +9.2% so với baseline
   - Tốc độ: ~3.7ms/ảnh

3. ✅ **Ứng dụng web hoàn chỉnh**
   - React + FastAPI
   - Hỗ trợ single và batch
   - Giao diện thân thiện

---

## SLIDE 20: HƯỚNG PHÁT TRIỂN

### Cải thiện dữ liệu

- Thu thập thêm dữ liệu cho classes yếu
- Tăng số lượng samples

### Cải thiện model

- Thử YOLOv8s/m
- Train thêm epochs (120-150)
- Fine-tuning

### Tính năng mới

- Video detection (real-time)
- Object tracking
- Export kết quả (JSON, CSV)
- Mobile app

---

## SLIDE 21: CẢM ƠN

# Cảm ơn đã lắng nghe!

## Câu hỏi & Thảo luận

**Sinh viên:** Phan Văn Tài  
**MSSV:** 2202081  
**Email:** [Email của bạn]

---

## HƯỚNG DẪN SỬ DỤNG SLIDE

### Số lượng slide: 21 slides

**Phân bố thời gian (cho 15 phút):**

1. **Slide 1-2**: Giới thiệu (1 phút)
2. **Slide 3-5**: Vấn đề & Lý thuyết (2 phút)
3. **Slide 6-10**: Xử lý dữ liệu ⭐ (5 phút) - **TẬP TRUNG**
4. **Slide 11-13**: Training & Kết quả (3 phút)
5. **Slide 14-18**: Demo ứng dụng ⭐ (3 phút) - **TẬP TRUNG**
6. **Slide 19-21**: Kết luận & Q&A (1 phút)

### Lưu ý khi thuyết trình

1. **Slide 6-10 (Xử lý dữ liệu)**: 
   - Giải thích chi tiết pipeline
   - Nhấn mạnh vấn đề imbalance
   - Show kết quả trước/sau

2. **Slide 14-18 (Demo)**:
   - **QUAN TRỌNG**: Chuẩn bị screenshots hoặc demo live
   - Giải thích từng tính năng
   - Show ví dụ kết quả thực tế

3. **Chuẩn bị**:
   - Screenshots giao diện web app
   - Ví dụ kết quả detection (ảnh trước/sau)
   - Có thể demo live nếu có thời gian

### Tips

- **Slide 7**: Có thể thêm biểu đồ phân bố classes
- **Slide 9**: Show bảng so sánh trước/sau rõ ràng
- **Slide 16-17**: **BẮT BUỘC** phải có screenshots
- **Slide 18**: Có thể show code example nếu cần

---

## NỘI DUNG CHI TIẾT CHO TỪNG SLIDE

### SLIDE 6-10: XỬ LÝ DỮ LIỆU (Chi tiết)

#### Slide 6: Tổng quan
- Show sơ đồ pipeline
- Giải thích từng bước

#### Slide 7: Phân tích
- **Thêm**: Biểu đồ phân bố classes (bar chart)
- Highlight vấn đề imbalance

#### Slide 8: Validation
- Giải thích tại sao cần validation
- Show số lượng samples bị loại bỏ

#### Slide 9: Cân bằng
- **QUAN TRỌNG**: Show bảng so sánh rõ ràng
- Giải thích oversampling strategy

#### Slide 10: Chia dữ liệu
- Giải thích stratified split
- Show cấu trúc dataset cuối cùng

### SLIDE 14-18: DEMO (Chi tiết)

#### Slide 14: Kiến trúc
- Sơ đồ kiến trúc rõ ràng
- Giải thích luồng dữ liệu

#### Slide 15: Tính năng
- List đầy đủ tính năng
- Highlight điểm nổi bật

#### Slide 16: Giao diện
- **BẮT BUỘC**: Screenshot giao diện
- Giải thích từng phần

#### Slide 17: Ví dụ kết quả
- **BẮT BUỘC**: Screenshots kết quả
- So sánh input/output

#### Slide 18: API
- Show code example (nếu có thời gian)
- Giải thích endpoints

---

**Tổng kết**: 21 slides, tập trung vào xử lý dữ liệu (5 slides) và demo (5 slides)

