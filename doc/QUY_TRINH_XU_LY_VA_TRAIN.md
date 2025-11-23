# QUY TRÌNH XỬ LÝ DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH

## PHẦN 1: XỬ LÝ DỮ LIỆU (data_preparation_pro.py)

### Tổng quan
Pipeline xử lý dữ liệu chuyên nghiệp để chuẩn bị dataset cho YOLOv8 training, tập trung vào việc xử lý data imbalance và đảm bảo chất lượng dữ liệu.

### Các bước thực hiện:

#### **BƯỚC 1: PHÂN TÍCH DATASET**
- Đếm số lượng ảnh theo từng class
- Tính toán imbalance ratio (tỷ lệ mất cân bằng)
- Phân loại classes thành 3 nhóm:
  - 🔴 Rất ít (< 15 ảnh): Loại bỏ
  - 🟡 Ít (15-30 ảnh): Cần oversampling
  - 🟢 Tốt (≥ 30 ảnh): Giữ nguyên
- **Kết quả**: Dataset gốc có 29,071 ảnh, 80 classes, imbalance ratio 73:1

#### **BƯỚC 2: THU THẬP & LỌC DỮ LIỆU**
- **Validate ảnh**: 
  - Kiểm tra format (.jpg, .jpeg, .png, .bmp)
  - Kiểm tra kích thước (32x32 ≤ size ≤ 10000x10000)
  - Kiểm tra corrupt images
- **Validate bounding boxes**:
  - Swap nếu tọa độ sai (x_min > x_max)
  - Clamp về phạm vi hợp lệ [0, img_width/height]
  - Loại bỏ nếu: width/height < 5 pixels hoặc area < 0.05% hoặc > 98% diện tích ảnh
- **Chuyển đổi sang YOLO format**:
  - Từ absolute coordinates → normalized coordinates [0, 1]
  - Format: `class_id center_x center_y width height`
- **Kết quả**: Thu thập được các samples hợp lệ từ train và test set

#### **BƯỚC 3: CÂN BẰNG DATASET**
- **Oversampling**: Các classes có 15-30 ảnh được tăng lên tối thiểu 30 ảnh bằng cách random copy
- **Loại bỏ**: Classes có < 15 ảnh (không có class nào trong dataset này)
- **Kết quả**: 
  - Squid: 22 → 30 samples
  - Turtle: 27 → 30 samples
  - Tổng: 28,184 samples (giảm từ 29,071 do loại bỏ samples không hợp lệ)

#### **BƯỚC 4: STRATIFIED TRAIN/VAL SPLIT**
- Chia dataset theo tỷ lệ 80/20 (Train/Validation)
- **Stratified**: Đảm bảo tỷ lệ classes giữ nguyên giữa train và val
- **Kết quả**:
  - Train: 22,518 samples (80%)
  - Validation: 5,666 samples (20%)

#### **BƯỚC 5: LƯU DỮ LIỆU**
- Tạo cấu trúc thư mục YOLO format:
  ```
  yolo_dataset_pro/
  ├── images/train/ (22,518 images)
  ├── images/val/ (5,666 images)
  ├── labels/train/ (22,518 label files)
  ├── labels/val/ (5,666 label files)
  └── data.yaml (config file)
  ```
- Tạo file `data.yaml` với thông tin: path, train/val paths, số classes (80), danh sách class names

---

## PHẦN 2: HUẤN LUYỆN MÔ HÌNH (model_training_optimized.py)

### Tổng quan
Training YOLOv8n với cấu hình tối ưu cho dataset đã được cân bằng, tập trung vào việc đạt mAP50 cao nhất với tốc độ hợp lý.

### Các bước thực hiện:

#### **BƯỚC 1: KIỂM TRA PHẦN CỨNG**
- Kiểm tra GPU (Tesla P100-PCIE-16GB, 16GB VRAM)
- Xác định device (CUDA/CPU)
- Đưa ra khuyến nghị batch size dựa trên VRAM

#### **BƯỚC 2: LOAD MODEL**
- Load YOLOv8n pretrained weights
- Đọc config từ `data.yaml`
- Xác nhận dataset: 80 classes, 22,518 train + 5,666 val

#### **BƯỚC 3: CẤU HÌNH TRAINING**

**Hyperparameters chính:**
- **Model**: YOLOv8n (nano) - 3.15M parameters, 8.7 GFLOPs
- **Epochs**: 100
- **Batch size**: 32 (phù hợp với VRAM 16GB)
- **Image size**: 640x640
- **Optimizer**: SGD (ổn định hơn AdamW cho balanced data)
- **Learning rate**: 
  - Initial (lr0): 0.002 (cao hơn mặc định để hội tụ nhanh)
  - Final (lrf): 0.0001
  - Scheduler: Cosine Annealing
- **Warmup**: 3 epochs

**Loss weights:**
- Box loss: 7.5
- Classification loss: 0.5 (giảm vì data đã balanced)
- DFL loss: 1.5

**Data Augmentation (vừa phải vì data đã balanced):**
- HSV: Hue ±0.015, Saturation ±0.7, Value ±0.4
- Geometric: Rotation ±8°, Translation ±10%, Scale 0.7-1.3, Shear ±2°
- Advanced: Mosaic 1.0 (tắt sau epoch 85), Mixup 0.1, Copy-paste 0.05, Flip 0.5

**Training strategies:**
- Early stopping: Patience = 40 epochs
- Save period: 5 epochs
- AMP (Automatic Mixed Precision): Enabled
- Close mosaic: Epoch 85

#### **BƯỚC 4: TRAINING**
- Thời gian training: ~8 giờ 21 phút
- Tốc độ: ~2.6-2.9 iterations/second
- Số iterations: 70,400 (704 batches/epoch × 100 epochs)

**Quá trình hội tụ:**
- Epoch 1: mAP50 = 0.124
- Epoch 10: mAP50 = 0.561
- Epoch 20: mAP50 = 0.689
- Epoch 50: mAP50 = 0.747
- Epoch 100: mAP50 = 0.755

**Loss giảm:**
- Box loss: 1.248 → 0.594 (giảm 52.4%)
- Classification loss: 3.722 → 0.588 (giảm 84.2%)
- DFL loss: 1.547 → 1.137 (giảm 26.5%)

#### **BƯỚC 5: VALIDATION**
- Load best model (epoch 100)
- Validate trên validation set (5,666 samples)
- Tính toán metrics cuối cùng

**Kết quả cuối cùng:**
- **mAP50**: 0.7565 (75.65%)
- **mAP50-95**: 0.6322 (63.22%)
- **Precision**: 0.7140
- **Recall**: 0.7469
- **F1-Score**: 0.7301

**So sánh với baseline:**
- Imbalanced data: mAP50 = 0.6925
- Balanced data: mAP50 = 0.7565
- **Cải thiện: +9.2%**

---

## TÓM TẮT QUY TRÌNH

### Xử lý dữ liệu:
1. Phân tích → 2. Validate & Clean → 3. Balance (oversampling) → 4. Stratified Split → 5. Lưu YOLO format

### Training:
1. Kiểm tra hardware → 2. Load model → 3. Cấu hình hyperparameters → 4. Training 100 epochs → 5. Validation

### Kết quả:
- Dataset: 28,184 samples (balanced), 80 classes
- Model: YOLOv8n, mAP50 = 0.7565
- Cải thiện: +9.2% so với baseline
- Thời gian: ~8.5 giờ training

---

## ĐIỂM NỔI BẬT CỦA QUY TRÌNH

1. **Xử lý imbalance chuyên nghiệp**: Phân tích, validate, và oversampling có hệ thống
2. **Stratified split**: Đảm bảo tỷ lệ classes giữ nguyên giữa train/val
3. **Augmentation vừa phải**: Tránh overfitting với data đã balanced
4. **Hyperparameters tối ưu**: SGD, LR cao hơn, loss weights phù hợp
5. **Early stopping**: Tránh overfitting, patience = 40 epochs

