# BÁO CÁO ĐỒ ÁN
## HỆ THỐNG NHẬN DIỆN ĐỐI TỢNG SỬ DỤNG YOLOv8 VỚI DEEP SORT TRACKING

**Sinh viên:** Phan Văn Tài - MSSV: 2202081  
**Giảng viên hướng dẫn:** Tiến sĩ Trần Ngọc Anh  
**Trường Đại học Tân Tạo - Khoa Công nghệ Thông tin**

---

## MỤC LỤC

1. [Tóm tắt (Abstract)](#1-tóm-tắt-abstract)
2. [Giới thiệu đề tài](#2-giới-thiệu-đề-tài)
3. [Cơ sở lý thuyết](#3-cơ-sở-lý-thuyết)
4. [Phân tích yêu cầu](#4-phân-tích-yêu-cầu)
5. [Thiết kế hệ thống](#5-thiết-kế-hệ-thống)
6. [Chuẩn bị dữ liệu](#6-chuẩn-bị-dữ-liệu)
7. [Huấn luyện mô hình](#7-huấn-luyện-mô-hình)
8. [Kết quả](#8-kết-quả)
9. [Demo / Ứng dụng](#9-demo--ứng-dụng)
10. [Đánh giá & Thảo luận](#10-đánh-giá--thảo-luận)
11. [Kết luận & Hướng phát triển](#11-kết-luận--hướng-phát-triển)
12. [Tài liệu tham khảo](#12-tài-liệu-tham-khảo)

---

## 1. TÓM TẮT (ABSTRACT)

Hệ thống nhận diện đối tượng real-time là một ứng dụng quan trọng trong nhiều lĩnh vực như giám sát an ninh, hỗ trợ người khiếm thị, và tự động hóa. Tuy nhiên, việc xây dựng một hệ thống hiệu quả đòi hỏi giải quyết nhiều thách thức, đặc biệt là vấn đề mất cân bằng dữ liệu (data imbalance) trong các dataset phổ biến như COCO.

Nghiên cứu này trình bày một pipeline hoàn chỉnh để xử lý và cân bằng dataset COCO 2014, từ 82,081 ảnh training ban đầu xuống còn 10,030 ảnh được chọn lọc thông qua thuật toán smart sampling dựa trên quality score. Dataset sau xử lý đạt được sự cân bằng với khoảng 250 ảnh cho mỗi class trong 80 classes, giảm tỷ lệ imbalance từ 321:1 xuống còn ~1:1.

Hệ thống được xây dựng sử dụng YOLOv8s (Small variant) với 11.2 triệu tham số và 28.8 GFLOPs, được huấn luyện trên GPU Tesla P100 với 17GB VRAM trong 120 epochs, mất khoảng 8 giờ. Mô hình được tối ưu hóa đặc biệt cho GPU P100 với batch size 28, 20 workers cho data loading, và Mixed Precision Training (AMP) để tăng tốc độ huấn luyện.

Kết quả thực nghiệm cho thấy mô hình đạt được mAP50 = 66.01%, Precision = 72.83%, Recall = 59.33%, và F1-Score = 65.40% trên validation set gồm 9,217 ảnh. Mặc dù kết quả này thấp hơn baseline (69.25%) khoảng 4.7%, nghiên cứu đã phân tích và đưa ra các nguyên nhân có thể như model quá lớn so với dataset size, dẫn đến overfitting tiềm ẩn.

Hệ thống web application được phát triển với kiến trúc Frontend (React) và Backend (FastAPI), tích hợp DeepSORT tracking để duy trì track IDs ổn định qua các frames. Đặc biệt, hệ thống hỗ trợ Text-to-Speech bằng tiếng Việt, gom kết quả theo lớp (ví dụ: "Phát hiện 2 xe tải. Phát hiện 1 người"), hữu ích cho người khiếm thị.

Nghiên cứu đóng góp một pipeline xử lý dữ liệu thông minh với quality scoring, phương pháp tối ưu hóa training cho GPU P100, và một hệ thống ứng dụng hoàn chỉnh với tính năng accessibility. Kết quả nghiên cứu cung cấp insights quan trọng về mối quan hệ giữa model size và dataset size, gợi ý rằng YOLOv8n (nano variant) có thể phù hợp hơn cho dataset 10k ảnh.

**Từ khóa:** Object Detection, YOLOv8, DeepSORT, Data Imbalance, Smart Sampling, Real-time Tracking, Web Application

---

## 2. GIỚI THIỆU ĐỀ TÀI

### 2.1. Bối cảnh và động lực

Nhận diện đối tượng (Object Detection) là một trong những bài toán cơ bản và quan trọng nhất trong lĩnh vực Computer Vision. Khác với bài toán phân loại ảnh (Image Classification), object detection không chỉ xác định xem đối tượng nào có trong ảnh mà còn phải xác định chính xác vị trí của chúng thông qua các bounding boxes. Ứng dụng của object detection rất đa dạng, từ giám sát an ninh, tự động hóa giao thông, hỗ trợ người khiếm thị, đến các hệ thống robot và xe tự lái.

Trong những năm gần đây, các mô hình deep learning, đặc biệt là YOLO (You Only Look Once) series, đã đạt được những thành tựu đáng kể trong việc nhận diện đối tượng real-time. YOLOv8, phiên bản mới nhất của series này, được phát triển bởi Ultralytics, mang lại hiệu suất cao với tốc độ xử lý nhanh, phù hợp cho các ứng dụng thời gian thực.

Tuy nhiên, việc huấn luyện các mô hình object detection hiệu quả đòi hỏi dataset chất lượng cao và cân bằng. Dataset COCO (Common Objects in Context) là một trong những dataset phổ biến nhất, nhưng nó có vấn đề mất cân bằng nghiêm trọng giữa các classes. Ví dụ, class "person" có hơn 7,000 ảnh trong khi class "toaster" chỉ có khoảng 25 ảnh, tạo ra tỷ lệ imbalance lên đến 321:1. Sự mất cân bằng này có thể dẫn đến việc model học tốt các classes phổ biến nhưng kém hiệu quả với các classes hiếm.

Ngoài ra, việc xây dựng một hệ thống ứng dụng hoàn chỉnh với khả năng tracking đối tượng qua nhiều frames và hỗ trợ người dùng khiếm thị là một thách thức kỹ thuật quan trọng. DeepSORT (Deep Simple Online and Realtime Tracking) là một giải pháp phổ biến cho bài toán multi-object tracking, kết hợp Kalman Filter và feature matching để duy trì track IDs ổn định.

### 2.2. Mục tiêu nghiên cứu

Nghiên cứu này đặt ra các mục tiêu chính sau:

1. **Xây dựng pipeline xử lý dữ liệu thông minh**: Phát triển một hệ thống tự động để chuyển đổi dataset COCO 2014 từ trạng thái mất cân bằng (321:1) sang trạng thái cân bằng (~1:1) thông qua thuật toán smart sampling dựa trên quality score.

2. **Huấn luyện mô hình YOLOv8 hiệu quả**: Tối ưu hóa quá trình training YOLOv8s trên balanced dataset, đặc biệt tối ưu cho GPU Tesla P100 với 17GB VRAM, đạt được mAP50 mục tiêu kỳ vọng từ 0.78-0.83 (mục tiêu tối thiểu ≥ 0.75). *Lưu ý: Kết quả thực tế đạt mAP50 = 0.6601, thấp hơn mục tiêu kỳ vọng, sẽ được phân tích chi tiết trong phần đánh giá.*

3. **Xây dựng hệ thống web application**: Phát triển một ứng dụng web hoàn chỉnh với khả năng:
   - Nhận diện đối tượng real-time từ camera
   - Upload và xử lý ảnh tĩnh
   - Tracking đối tượng với DeepSORT
   - Audio feedback bằng tiếng Việt cho người khiếm thị

4. **Đánh giá và phân tích kết quả**: So sánh hiệu suất của mô hình trên balanced dataset với baseline, phân tích các yếu tố ảnh hưởng đến kết quả, và đưa ra các đề xuất cải thiện.

### 2.3. Phạm vi nghiên cứu

Nghiên cứu này tập trung vào:

- **Dataset**: COCO 2014 với 80 classes, xử lý từ 82,081 training images xuống còn 10,030 images balanced
- **Model**: YOLOv8s (Small variant) với 11.2M parameters
- **Hardware**: GPU Tesla P100-PCIE-16GB (17.06 GB VRAM)
- **Tracking**: DeepSORT với Kalman Filter và histogram-based feature extraction
- **Application**: Web application với React frontend và FastAPI backend

Nghiên cứu không bao gồm:
- Các mô hình object detection khác ngoài YOLOv8
- Các phương pháp tracking khác ngoài DeepSORT
- Mobile application hoặc edge deployment
- Video processing batch

### 2.4. Cấu trúc báo cáo

Báo cáo được tổ chức thành 12 chương chính:

- **Chương 1**: Tóm tắt nghiên cứu
- **Chương 2**: Giới thiệu đề tài (chương này)
- **Chương 3**: Cơ sở lý thuyết về YOLOv8, DeepSORT, và các metrics đánh giá
- **Chương 4**: Phân tích yêu cầu chức năng và phi chức năng
- **Chương 5**: Thiết kế kiến trúc hệ thống và lựa chọn model
- **Chương 6**: Chi tiết pipeline xử lý dữ liệu và smart sampling
- **Chương 7**: Quá trình huấn luyện mô hình và tối ưu hóa cho P100
- **Chương 8**: Kết quả thực nghiệm và phân tích
- **Chương 9**: Demo hệ thống và các tính năng
- **Chương 10**: Đánh giá, thảo luận và lessons learned
- **Chương 11**: Kết luận và hướng phát triển
- **Chương 12**: Tài liệu tham khảo

---

## 3. CƠ SỞ LÝ THUYẾT

### 3.1. Object Detection và YOLO

#### 3.1.1. Tổng quan về Object Detection

Object Detection là bài toán xác định vị trí và phân loại các đối tượng trong ảnh. Khác với Image Classification chỉ trả về nhãn của toàn bộ ảnh, object detection phải:
- Xác định số lượng đối tượng trong ảnh
- Vẽ bounding box cho mỗi đối tượng
- Phân loại từng đối tượng vào một trong các classes

Các phương pháp object detection truyền thống như R-CNN, Fast R-CNN, Faster R-CNN sử dụng two-stage approach: đầu tiên tạo region proposals, sau đó phân loại và refine bounding boxes. Tuy nhiên, các phương pháp này thường chậm và không phù hợp cho real-time applications.

#### 3.1.2. YOLO (You Only Look Once)

YOLO được giới thiệu lần đầu vào năm 2016 bởi Redmon et al., với ý tưởng cách mạng: thay vì tạo region proposals, YOLO chia ảnh thành grid cells và mỗi cell dự đoán trực tiếp bounding boxes và class probabilities. Điều này làm cho YOLO nhanh hơn đáng kể so với các phương pháp two-stage.

YOLO đã trải qua nhiều phiên bản:
- **YOLOv1** (2016): Kiến trúc đơn giản, tốc độ cao nhưng độ chính xác thấp
- **YOLOv2/YOLO9000** (2017): Cải thiện accuracy với anchor boxes
- **YOLOv3** (2018): Multi-scale detection với 3 scales
- **YOLOv4** (2020): Tối ưu hóa kiến trúc và training strategy
- **YOLOv5** (2020): PyTorch implementation, dễ sử dụng
- **YOLOv8** (2023): Phiên bản mới nhất với nhiều cải tiến

#### 3.1.3. YOLOv8 Architecture

YOLOv8 được phát triển bởi Ultralytics với kiến trúc được tối ưu hóa cho cả accuracy và speed. Kiến trúc chính bao gồm:

**Backbone - CSPDarknet:**
- Sử dụng Cross Stage Partial (CSP) connections để giảm computational cost
- C2f module thay thế C3 module trong YOLOv5, cải thiện gradient flow
- Efficient feature extraction với depthwise separable convolutions

**Neck - PANet (Path Aggregation Network):**
- Kết hợp top-down và bottom-up paths để tận dụng cả high-level và low-level features
- Feature Pyramid Network (FPN) để xử lý objects ở nhiều scales khác nhau

**Head - Decoupled Head:**
- Tách riêng classification và regression tasks
- Giảm conflict giữa classification và localization
- Cải thiện accuracy đáng kể

**Loss Functions:**
- **Box Loss**: DFL (Distribution Focal Loss) thay vì IoU loss, cải thiện localization accuracy
- **Class Loss**: BCE (Binary Cross Entropy) với label smoothing
- **DFL Loss**: Giúp model học phân phối xác suất của bounding box coordinates

**Non-Maximum Suppression (NMS):**
- Loại bỏ các detections trùng lặp
- IoU threshold để quyết định các boxes nào được giữ lại
- Mặc định IoU threshold = 0.45

#### 3.1.4. YOLOv8 Variants và So sánh

YOLOv8 có 5 variants với kích thước và độ phức tạp khác nhau:

| Variant | Parameters | GFLOPs | mAP50 (COCO) | Speed (ms) |
|---------|-----------|--------|--------------|------------|
| YOLOv8n (nano) | 3.2M | 8.1 | ~37.3 | ~0.99 |
| YOLOv8s (small) | 11.2M | 28.8 | ~44.9 | ~1.20 |
| YOLOv8m (medium) | 25.9M | 78.9 | ~50.2 | ~1.83 |
| YOLOv8l (large) | 43.7M | 165.2 | ~52.9 | ~2.39 |
| YOLOv8x (xlarge) | 68.2M | 257.8 | ~53.9 | ~3.53 |

**Lựa chọn YOLOv8s cho nghiên cứu:**
- **Lý do**: Mục tiêu đạt mAP50 = 0.78-0.83, cần model có capacity đủ lớn
- **Trade-off**: YOLOv8s có 11.2M parameters, lớn hơn YOLOv8n (3.2M) nhưng nhỏ hơn YOLOv8m (25.9M)
- **Kết quả thực tế**: mAP50 = 0.6601, thấp hơn kỳ vọng, có thể do model quá lớn so với dataset size (10k images)

**So sánh với YOLOv8n:**
- YOLOv8n phù hợp hơn cho dataset nhỏ (< 20k images)
- YOLOv8s có thể dẫn đến overfitting với dataset 10k images
- Đề xuất: Thử YOLOv8n trong tương lai để so sánh

### 3.2. Multi-Object Tracking với DeepSORT

#### 3.2.1. Tổng quan về Tracking

Multi-Object Tracking (MOT) là bài toán theo dõi nhiều đối tượng qua nhiều frames trong video. Khác với detection chỉ xử lý từng frame độc lập, tracking phải:
- Duy trì identity của mỗi đối tượng qua frames
- Xử lý các trường hợp: xuất hiện mới, biến mất, occlusion
- Xử lý các trường hợp: ID switching, fragmentation

#### 3.2.2. DeepSORT Architecture

DeepSORT (Deep Simple Online and Realtime Tracking) là một phương pháp tracking phổ biến, kết hợp:

**Kalman Filter:**
- Dự đoán vị trí của đối tượng ở frame tiếp theo
- State vector: [x, y, s, r, x', y', s'] với:
  - x, y: center coordinates
  - s: scale (area)
  - r: aspect ratio
  - x', y', s': velocities
- Update state khi có detection mới

**Feature Extraction:**
- Trong nghiên cứu này, sử dụng histogram-based features (128 dimensions)
- Extract từ bounding box region: color histograms (B, G, R channels)
- Normalize thành unit vector để tính cosine similarity

**Association:**
- Tính cost matrix kết hợp IoU distance và feature distance
- Weighted combination: 0.5 × IoU_cost + 0.5 × Feature_cost
- Hungarian algorithm để tìm optimal matching
- Threshold để filter matches có cost quá cao

**Track Management:**
- **New tracks**: Tạo track mới cho unmatched detections
- **Confirmed tracks**: Tracks đã match ≥ min_hits (default: 3) frames
- **Deleted tracks**: Tracks không match trong max_age (default: 30) frames

### 3.3. Dataset và Metrics

#### 3.3.1. COCO Dataset

COCO (Common Objects in Context) là một trong những dataset phổ biến nhất cho object detection:
- **COCO 2014**: 82,081 training labels được sử dụng trong nghiên cứu này, 40,504 validation images
- **80 classes**: person, bicycle, car, ..., toothbrush
- **Format**: YOLO format với normalized coordinates (x_center, y_center, width, height)

**Vấn đề Imbalance:**
- Class "person": 7,418 images
- Class "toaster": 25 images
- Tỷ lệ imbalance: 321:1
- Ảnh hưởng: Model học tốt classes phổ biến, kém với classes hiếm

#### 3.3.2. Evaluation Metrics

**mAP (mean Average Precision):**
- **mAP50**: Average Precision với IoU threshold = 0.5
- **mAP50-95**: Average Precision với IoU từ 0.5 đến 0.95 (step 0.05)
- Tính cho mỗi class, sau đó lấy trung bình

**Precision:**
- Tỷ lệ detections đúng trong tổng số detections
- Precision = TP / (TP + FP)
- Trong nghiên cứu: 0.7283 (72.83%)

**Recall:**
- Tỷ lệ ground truth được detect đúng
- Recall = TP / (TP + FN)
- Trong nghiên cứu: 0.5933 (59.33%)

**F1-Score:**
- Harmonic mean của Precision và Recall
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Trong nghiên cứu: 0.6540 (65.40%)
- Đánh giá sự cân bằng giữa Precision và Recall

**IoU (Intersection over Union):**
- Đo độ overlap giữa predicted box và ground truth box
- IoU = Area of Intersection / Area of Union
- Threshold = 0.5 để quyết định TP/FP

---

## 4. PHÂN TÍCH YÊU CẦU

### 4.1. Yêu cầu chức năng

Hệ thống cần đáp ứng các yêu cầu chức năng sau:

#### 4.1.1. Real-time Object Detection từ Camera
- **Mô tả**: Nhận diện đối tượng real-time từ webcam hoặc camera device
- **Yêu cầu kỹ thuật**:
  - Frame rate ≥ 10 FPS để đảm bảo trải nghiệm mượt mà
  - Latency < 200ms per frame
  - Hỗ trợ nhiều resolutions (640×480, 1280×720)
- **Input**: Video stream từ camera
- **Output**: Ảnh với bounding boxes, class labels, confidence scores

#### 4.1.2. Image Upload và Detection
- **Mô tả**: Cho phép người dùng upload ảnh tĩnh để nhận diện
- **Yêu cầu kỹ thuật**:
  - Hỗ trợ formats: JPG, PNG, BMP, WEBP, TIFF
  - Max file size: 10MB
  - Drag & drop interface
- **Input**: Image file từ user
- **Output**: Ảnh với bounding boxes, bảng kết quả chi tiết, statistics

#### 4.1.3. Multi-Object Tracking
- **Mô tả**: Duy trì track IDs ổn định cho các đối tượng qua nhiều frames
- **Yêu cầu kỹ thuật**:
  - Track ID không thay đổi khi đối tượng di chuyển
  - Xử lý occlusion (đối tượng bị che khuất)
  - Xử lý ID switching (tránh nhầm lẫn giữa các đối tượng)
- **Input**: Sequence of frames với detections
- **Output**: Tracks với stable IDs, history paths

#### 4.1.4. Audio Feedback
- **Mô tả**: Phát âm kết quả detection bằng tiếng Việt cho người khiếm thị
- **Yêu cầu kỹ thuật**:
  - Text-to-Speech (TTS) sử dụng Web Speech API
  - Gom kết quả theo lớp (ví dụ: "Phát hiện 2 xe tải. Phát hiện 1 người")
  - Chỉ đọc đối tượng mới trong camera mode (dựa trên track IDs)
  - Hỗ trợ bật/tắt audio, đọc lại
- **Input**: Detection results
- **Output**: Audio speech bằng tiếng Việt

#### 4.1.5. Visualization và Statistics
- **Mô tả**: Hiển thị kết quả detection với bounding boxes và thống kê
- **Yêu cầu kỹ thuật**:
  - Vẽ bounding boxes với màu sắc theo confidence
  - Hiển thị class name, confidence score, track ID
  - Bảng kết quả có thể sắp xếp theo confidence/class
  - Statistics: tổng số objects, classes, avg confidence
- **Input**: Detection results
- **Output**: Visualized image, results table, statistics

### 4.2. Yêu cầu phi chức năng

#### 4.2.1. Performance
- **Frame Rate**: ≥ 10 FPS trong camera mode
- **Latency**: < 200ms per frame
- **Throughput**: Xử lý được nhiều requests đồng thời
- **Memory**: Sử dụng < 4GB GPU memory

#### 4.2.2. Accuracy
- **mAP50**: 
  - Mục tiêu tối thiểu: ≥ 0.75 (75%)
  - Mục tiêu kỳ vọng: 0.78-0.83 (dựa trên balanced dataset và model YOLOv8s)
- **Precision**: ≥ 0.70 (70%)
- **Recall**: ≥ 0.65 (65%)
- **F1-Score**: ≥ 0.65 (65%) - đánh giá sự cân bằng giữa Precision và Recall
- **Kết quả thực tế**: mAP50 = 0.6601, Precision = 0.7283, Recall = 0.5933, F1-Score = 0.6540
  - *Precision đạt mục tiêu, F1-Score gần đạt mục tiêu, nhưng mAP50 và Recall thấp hơn mục tiêu, sẽ được phân tích trong phần đánh giá*

#### 4.2.3. Usability
- **User Interface**: Responsive, dễ sử dụng, hỗ trợ mobile
- **Accessibility**: Hỗ trợ người khiếm thị với audio feedback
- **Localization**: Giao diện và kết quả hoàn toàn bằng tiếng Việt
- **Error Handling**: Thông báo lỗi rõ ràng, user-friendly

#### 4.2.4. Scalability
- **Concurrent Users**: Hỗ trợ nhiều sessions đồng thời
- **Session Management**: Tự động cleanup sessions không hoạt động (timeout 5 phút)
- **Resource Management**: Tối ưu memory và GPU usage

#### 4.2.5. Security
- **File Upload**: Validate file type, size, sanitize filenames
- **Path Traversal Protection**: Ngăn chặn directory traversal attacks
- **CORS Configuration**: Cấu hình CORS để bảo mật API
- **Input Validation**: Validate thresholds, parameters

### 4.3. Yêu cầu kỹ thuật

#### 4.3.1. Backend
- **Framework**: FastAPI (Python 3.8+)
- **Model**: YOLOv8s (Ultralytics)
- **Tracking**: DeepSORT implementation
- **Dependencies**: 
  - ultralytics==8.3.223
  - opencv-python==4.8.1.78
  - scipy>=1.9.0 (cho Hungarian algorithm)
  - filterpy>=1.4.5 (cho Kalman Filter)

#### 4.3.2. Frontend
- **Framework**: React 18.2.0
- **Styling**: Tailwind CSS 3.3.6
- **HTTP Client**: Axios 1.6.0
- **Audio**: Web Speech API (SpeechSynthesis)

#### 4.3.3. Hardware
- **GPU**: NVIDIA Tesla P100-PCIE-16GB (17.06 GB VRAM)
- **CPU**: Multi-core processor
- **Memory**: ≥ 8GB RAM
- **Storage**: ≥ 10GB cho dataset và model

#### 4.3.4. Software
- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.8+
- **Node.js**: 14+ (khuyến nghị 16+)
- **CUDA**: 11.0+ (cho GPU support)

---

## 5. THIẾT KẾ HỆ THỐNG

### 5.1. Kiến trúc tổng thể

Hệ thống được thiết kế theo kiến trúc 3 tầng (3-tier architecture) với sự tách biệt rõ ràng giữa Presentation Layer, Application Layer, và Data/Model Layer.

**Tầng 1 - Frontend (Presentation Layer):**
- **Framework**: React 18.2.0 với functional components và hooks
- **Styling**: Tailwind CSS 3.3.6 cho responsive design
- **State Management**: React useState, useEffect, useRef hooks
- **HTTP Client**: Axios 1.6.0 để giao tiếp với backend
- **Audio**: Web Speech API (SpeechSynthesis) cho Text-to-Speech

**Tầng 2 - Backend (Application Layer):**
- **Framework**: FastAPI 0.104.1 (Python async web framework)
- **API Server**: Uvicorn với ASGI
- **CORS**: Cấu hình CORS middleware để cho phép frontend kết nối
- **Session Management**: Quản lý tracking sessions với timeout 5 phút
- **File Handling**: Temporary file storage với auto-cleanup

**Tầng 3 - Model Layer:**
- **Detection Model**: YOLOv8s (Ultralytics) - 11.2M parameters
- **Tracking**: DeepSORT implementation với Kalman Filter
- **Feature Extraction**: Histogram-based features (128 dimensions)

**Luồng xử lý chính:**

1. **Image Upload Mode:**
   ```
   User uploads image → Frontend (React) → Backend API (/api/detect) 
   → ObjectDetector.detect_image() → YOLOv8 inference 
   → Extract detections → Format response → Frontend displays results
   ```

2. **Camera Mode:**
   ```
   Camera stream → Frontend captures frame → Backend API (/api/detect-video)
   → VideoTracker.process_frame() → YOLOv8 detection + DeepSORT tracking
   → Update tracks → Format response → Frontend displays with track IDs
   ```

### 5.2. Lựa chọn Model

#### 5.2.1. Phân tích Dataset Size

Dataset sau xử lý có:
- **Training images**: 10,030 ảnh
- **Validation images**: 9,217 ảnh
- **Total**: 19,247 ảnh
- **Classes**: 80 classes
- **Avg per class**: ~125 ảnh (sau khi balanced)

Đây là một dataset tương đối nhỏ so với các dataset lớn như COCO full (82k+ images). Với dataset size này, việc lựa chọn model size là rất quan trọng.

#### 5.2.2. So sánh YOLOv8n vs YOLOv8s

**YOLOv8n (Nano):**
- Parameters: 3.2M
- GFLOPs: 8.1
- Ưu điểm:
  - Nhẹ, nhanh
  - Phù hợp với dataset nhỏ (< 20k images)
  - Ít khả năng overfitting
- Nhược điểm:
  - Capacity thấp, có thể không đủ cho 80 classes
  - mAP thấp hơn trên COCO full dataset

**YOLOv8s (Small):**
- Parameters: 11.2M
- GFLOPs: 28.8
- Ưu điểm:
  - Capacity cao hơn, phù hợp cho nhiều classes
  - Có thể đạt mAP cao hơn với dataset tốt
- Nhược điểm:
  - Nặng hơn, chậm hơn
  - Có thể overfitting với dataset nhỏ

#### 5.2.3. Quyết định lựa chọn

**Lựa chọn: YOLOv8s**

**Lý do:**
1. **Mục tiêu mAP cao**: Mục tiêu đạt mAP50 = 0.78-0.83, cần model có capacity đủ lớn
2. **80 classes**: Số lượng classes lớn (80) đòi hỏi model có đủ capacity để học
3. **Balanced dataset**: Dataset đã được balanced, có thể tận dụng được capacity của model lớn hơn
4. **GPU đủ mạnh**: Tesla P100 với 17GB VRAM có thể handle YOLOv8s với batch size hợp lý

**Kết quả thực tế:**
- mAP50 = 0.6601 (thấp hơn baseline 0.6925)
- Có thể do model quá lớn so với dataset size → overfitting tiềm ẩn
- **Lesson learned**: Nên thử YOLOv8n để so sánh trong tương lai

#### 5.2.4. Đề xuất cải thiện

Dựa trên kết quả, đề xuất:
1. **Thử YOLOv8n**: Với dataset 10k images, YOLOv8n có thể phù hợp hơn
2. **Fine-tuning**: Nếu dùng YOLOv8s, cần điều chỉnh hyperparameters (LR, augmentation, regularization)
3. **Data augmentation mạnh hơn**: Tăng augmentation để tăng effective dataset size

### 5.3. Backend Design

#### 5.3.1. API Endpoints

**GET /api/model-info**
- Trả về thông tin model: số classes, danh sách classes, default thresholds
- Response: JSON với model metadata

**POST /api/detect**
- Nhận diện đối tượng trong ảnh tĩnh
- Input: Multipart form data (file, conf_threshold, iou_threshold)
- Output: JSON với detections, image_base64, statistics
- Timeout: 30 seconds

**POST /api/detect-video**
- Nhận diện và tracking đối tượng trong video frame
- Input: Multipart form data (file, conf_threshold, iou_threshold, session_id)
- Output: JSON với tracks, image_base64, statistics, session_id
- Timeout: 30 seconds

**POST /api/reset-tracking-session**
- Reset tracking session (xóa tất cả tracks)
- Input: session_id
- Output: Success/failure message

**POST /api/compare-thresholds**
- So sánh kết quả detection với nhiều confidence thresholds khác nhau
- Input: Multipart form data (file, thresholds: JSON array như "[0.1, 0.25, 0.5, 0.75]")
- Output: JSON với comparisons cho mỗi threshold (count, classes)
- Timeout: 60 seconds
- Use case: Giúp người dùng chọn threshold phù hợp bằng cách so sánh số lượng detections và classes ở các mức threshold khác nhau
- Giới hạn: Tối đa 10 thresholds được phép so sánh trong một request

#### 5.3.2. ObjectDetector Class

Wrapper class cho YOLO model với các phương thức chính:

**Các phương thức:**
- `__init__()`: Khởi tạo với model path và thresholds (conf_threshold=0.25, iou_threshold=0.45)
- `detect_image()`: Nhận diện đối tượng trong một ảnh, trả về results và annotated image
- `detect_folder()`: Nhận diện tất cả ảnh trong một folder
- `compare_thresholds()`: So sánh kết quả với nhiều confidence thresholds khác nhau

**Chức năng:**
- Load YOLOv8 model từ file `best.pt`
- Thực hiện inference với configurable thresholds
- Trả về results và annotated images

#### 5.3.3. VideoTracker Class

Kết hợp YOLO detection và DeepSORT tracking với các phương thức:

**Các phương thức:**
- `__init__()`: Khởi tạo với model path và thresholds, tạo ObjectDetector và DeepSORT tracker
- `process_frame()`: Xử lý một frame: detect với YOLO, track với DeepSORT, trả về results và tracks
- `reset()`: Reset tracker, xóa tất cả tracks

**Chức năng:**
- YOLO detection cho mỗi frame
- DeepSORT tracking để duy trì track IDs
- Format tracks thành dict cho API response
- Vẽ bounding boxes với track IDs và colors

#### 5.3.4. Session Management

- **Session Storage**: Dictionary lưu trữ tracker instances theo session_id
- **Session Timeout**: 5 phút (300 seconds) - tự động cleanup
- **Session ID**: Format `session_{timestamp}_{random}` để đảm bảo unique
- **Cleanup**: Tự động xóa sessions không hoạt động

### 5.4. Frontend Design

#### 5.4.1. Component Architecture

**App.jsx (Main Component):**
- Quản lý mode: 'home', 'camera', 'image'
- State management: selectedFile, detections, modelInfo
- Audio service integration

**HomeView.jsx:**
- Màn hình chọn chế độ: Camera hoặc Image
- Navigation buttons

**CameraView.jsx:**
- Real-time camera stream với MediaDevices API
- Frame capture và gửi đến backend
- Bounding boxes overlay với track IDs
- Audio feedback cho đối tượng mới
- Performance optimization: frame skipping, image compression

**ImageUpload.jsx:**
- Drag & drop interface
- File validation (type, size)
- Preview image

**ImagePreview.jsx:**
- Hiển thị ảnh với bounding boxes
- Zoom & pan functionality

**ResultsTable.jsx:**
- Bảng kết quả với sortable columns
- Statistics display

#### 5.4.2. Audio Service

**audioService.js:**
- Text-to-Speech sử dụng Web Speech API
- Gom kết quả theo lớp (ví dụ: "Phát hiện 2 xe tải. Phát hiện 1 người")
- Chỉ đọc đối tượng mới trong camera mode (dựa trên track IDs)
- Queue management để tránh spam audio
- Debounce mechanism

**Tính năng:**
- `speakDetections()`: Phát âm nhiều detections, gom theo lớp
- `speakSystemMessage()`: Phát âm thông báo hệ thống
- `stop()`: Dừng audio hiện tại
- `setEnabled()`: Bật/tắt audio

#### 5.4.3. Performance Optimization

**Frame Skipping:**
- Skip frame nếu đang detect frame trước
- Tránh queue quá nhiều requests

**Image Compression:**
- Resize frame từ 640×480 xuống 320×240
- JPEG quality = 0.6 (giảm từ 0.8)
- Giảm file size ~75%, tăng tốc độ upload

**Request Queue:**
- Sử dụng AbortController để cancel requests cũ
- Tránh race conditions
- Chỉ giữ 1 request active tại một thời điểm

### 5.5. Security và Error Handling

#### 5.5.1. Security Features

- **Path Traversal Protection**: Sanitize filenames, chỉ cho phép alphanumeric và một số ký tự đặc biệt
- **File Size Validation**: Max 10MB per file
- **File Type Validation**: Chỉ cho phép image formats (JPG, PNG, BMP, WEBP, TIFF)
- **CORS Configuration**: Configurable via environment variable
- **Input Validation**: Validate thresholds (0-1 range)

#### 5.5.2. Error Handling

- **Timeout Handling**: 30s cho detection, 60s cho batch
- **GPU Error Handling**: Catch CUDA errors, fallback messages
- **Memory Error Handling**: Catch OOM errors, suggest smaller images
- **Network Error Handling**: Retry logic, user-friendly messages

---

## 6. CHUẨN BỊ DỮ LIỆU

### 6.1. Dataset gốc - COCO 2014

**Thống kê ban đầu:**
- **Training images**: 82,081 ảnh
- **Validation images**: 40,504 ảnh
- **Total images**: 122,585 ảnh
- **Classes**: 80 classes
- **Format**: YOLO format với normalized coordinates

**Vấn đề Imbalance nghiêm trọng:**

Bảng phân bố một số classes (top và bottom):

| Class | Số ảnh | Tỷ lệ |
|-------|--------|-------|
| person | 7,418 | 9.0% |
| car | 2,197 | 2.7% |
| bicycle | 713 | 0.9% |
| ... | ... | ... |
| toaster | 25 | 0.03% |
| hair drier | 15 | 0.02% |

**Tỷ lệ imbalance**: 321:1 (person : toaster)

**Ảnh hưởng:**
- Model học tốt classes phổ biến (person, car)
- Model kém với classes hiếm (toaster, hair drier)
- Precision và Recall không đồng đều giữa các classes

### 6.2. Pipeline xử lý dữ liệu

Pipeline được thiết kế thành 6 bước chính:

#### 6.2.1. Bước 1: Phân tích Dataset

**Mục tiêu**: Quét toàn bộ dataset để hiểu phân bố classes

**Quy trình:**
1. Load class names từ `coco.names` (80 classes)
2. Quét 82,081 training labels
3. Đếm số ảnh chứa mỗi class
4. Phân loại classes:
   - **Classes đủ** (≥250 ảnh): 78 classes
   - **Classes thiếu** (<250 ảnh): 2 classes

**Kết quả:**
- 78 classes có ≥250 ảnh
- 2 classes có <250 ảnh
- Tổng số ảnh unique: ~20,000 ảnh (một ảnh có thể chứa nhiều classes)

#### 6.2.2. Bước 2: Smart Sampling

**Mục tiêu**: Chọn 250 ảnh tốt nhất cho mỗi class (classes đủ)

**Thuật toán Quality Score:**

Mỗi ảnh được đánh giá bằng quality score dựa trên:

1. **Số classes trong ảnh** (weight: 3.0)
   - Ảnh chứa nhiều classes → score cao hơn
   - Lý do: Ảnh đa dạng giúp model học tốt hơn

2. **Bbox area** (weight: 2.0 hoặc 1.0)
   - Area 0.05-0.6: +2.0 (optimal size)
   - Area 0.01-0.05: +1.0 (small but acceptable)
   - Area <0.01 hoặc >0.6: +0.0 (too small hoặc too large)
   - Lý do: Bbox quá nhỏ hoặc quá lớn khó detect

3. **Vị trí bbox** (weight: 1.0)
   - Center (x, y) trong khoảng [0.2, 0.8]: +1.0
   - Lý do: Bbox ở trung tâm thường dễ detect hơn

**Công thức tính Quality Score:**
- score = (số_classes_trong_ảnh × 3.0) + bbox_area_score + bbox_position_score
- Trong đó:
  - bbox_area_score: +2.0 nếu area 0.05-0.6, +1.0 nếu area 0.01-0.05, +0.0 nếu khác
  - bbox_position_score: +1.0 nếu center (x,y) trong [0.2, 0.8], +0.0 nếu khác

**Quy trình:**
1. Với mỗi class đủ (≥250 ảnh):
   - Tính quality score cho tất cả ảnh chứa class đó
   - Sắp xếp theo score giảm dần
   - Chọn top 250 ảnh
2. Với mỗi class thiếu (<250 ảnh):
   - Giữ tất cả ảnh có sẵn

**Kết quả:**
- 9,809 ảnh được chọn từ smart sampling
- Mỗi class đủ có đúng 250 ảnh được chọn
- Classes thiếu giữ nguyên tất cả ảnh

#### 6.2.3. Bước 3: Augmentation

**Mục tiêu**: Tăng số lượng ảnh cho classes thiếu

**Kỹ thuật Augmentation:**
1. **Horizontal Flip**: Flip ảnh theo trục dọc, điều chỉnh bbox coordinates
2. **Brightness Adjustment**: Nhân pixel values với random factor [0.85, 1.15]

**Quy trình:**
1. Với mỗi class thiếu:
   - Tính số ảnh cần: `need = 250 - current_count`
   - Random chọn ảnh từ dataset hiện có
   - Áp dụng augmentation
   - Lưu ảnh mới với suffix `_aug{i}.jpg`
   - Điều chỉnh bbox coordinates cho flip

**Kết quả:**
- 221 ảnh được augment
- Tổng training images: 10,030 ảnh (9,809 + 221)

**Lưu ý:**
- Chỉ augment cho classes thiếu
- Validate bbox coordinates sau augmentation
- Loại bỏ bbox không hợp lệ (area < 0.01)

#### 6.2.4. Bước 4: Validation Processing

**Mục tiêu**: Tạo validation set balanced

**Quy trình:**
1. Quét 10,000 validation labels (giới hạn để tiết kiệm thời gian)
2. Với mỗi class:
   - Sample ngẫu nhiên tối đa 250 ảnh
3. Copy ảnh và labels tương ứng

**Kết quả:**
- 9,217 validation images
- Phân bố tương đối cân bằng giữa các classes

#### 6.2.5. Bước 5: Tạo Config

**Mục tiêu**: Tạo file `data.yaml` cho YOLO format

**Nội dung file data.yaml:**
- path: Đường dẫn đến dataset root
- train: Thư mục chứa training images (images/train)
- val: Thư mục chứa validation images (images/val)
- nc: Số lượng classes (80)
- names: Dictionary mapping class ID sang class name (0: person, 1: bicycle, 2: car, ...)

#### 6.2.6. Bước 6: Tổng kết

**Kết quả cuối cùng:**

| Metric | Giá trị |
|--------|---------|
| Training images | 10,030 |
| Validation images | 9,217 |
| Total images | 19,247 |
| Classes | 80 |
| Target per class | 250 |
| Augmented images | 221 |
| Imbalance ratio | ~1:1 (từ 321:1) |

**So sánh Before/After:**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Training images | 82,081 | 10,030 | -87.8% |
| Imbalance ratio | 321:1 | ~1:1 | -99.7% |
| Avg per class | Varies | ~250 | Balanced |

**Lưu ý quan trọng:**
- Tổng ảnh (10,030) < 80 × 250 = 20,000 vì một ảnh có thể chứa nhiều classes
- Điều này là **đúng** cho YOLO training - một ảnh có thể train nhiều classes cùng lúc
- Ví dụ: Ảnh có person + car sẽ được đếm cho cả 2 classes

### 6.3. Phân tích Dataset sau xử lý

**Phân bố classes (10 classes đầu):**

| Class ID | Class Name | Selected Images | Total Available |
|----------|------------|-----------------|-----------------|
| 0 | person | 250 | 7,418 |
| 1 | bicycle | 250 | 713 |
| 2 | car | 250 | 2,197 |
| 3 | motorbike | 250 | 400 |
| 4 | aeroplane | 250 | 255 |
| 5 | bus | 250 | 521 |
| 6 | train | 250 | 269 |
| 7 | truck | 250 | 1,155 |
| 8 | boat | 250 | 359 |
| 9 | traffic light | 250 | 599 |

**Đặc điểm:**
- Tất cả 80 classes đều có ≥250 ảnh (sau augmentation)
- Phân bố đồng đều giữa các classes
- Quality score đảm bảo chọn ảnh tốt nhất

---

## 7. HUẤN LUYỆN MÔ HÌNH

### 7.1. Môi trường Training

**Hardware:**
- **GPU**: Tesla P100-PCIE-16GB
- **VRAM**: 17.06 GB
- **CUDA**: Version 11.0+
- **CPU**: Multi-core processor

**Software:**
- **Python**: 3.10.12
- **PyTorch**: 2.0.0
- **Ultralytics**: 8.3.233
- **CUDA**: Enabled với cuDNN

**Dataset:**
- **Training**: 10,030 images
- **Validation**: 9,217 images
- **Classes**: 80
- **Format**: YOLO format

**Thời gian Training:**
- **Start**: 2025-11-30 11:30:41
- **End**: 2025-11-30 19:16:10
- **Duration**: ~8 giờ (7 giờ 45 phút)

### 7.2. Tối ưu hóa cho GPU P100

#### 7.2.1. Auto-detection và Configuration

Hệ thống tự động phát hiện GPU và tối ưu hóa:

**GPU Detection:**
- Tự động phát hiện CUDA availability
- Lấy GPU name: Tesla P100-PCIE-16GB
- Tính toán VRAM: 17.06 GB từ device properties

**Auto-tuning Batch Size:**
- **YOLOv8s**: Batch size = 28 (tự động tính toán)
- **Lý do**: Model 's' lớn hơn, cần giảm batch để fit VRAM
- **Trade-off**: Batch nhỏ hơn → training chậm hơn nhưng ổn định hơn

**Workers:**
- **Data Loading**: 20 workers (tối đa)
- **Lý do**: Tăng tốc độ load data, giảm bottleneck

**Cache:**
- **Image Cache**: Enabled
- **Lý do**: Cache images trong RAM để tăng tốc độ training

#### 7.2.2. Mixed Precision Training (AMP)

**Automatic Mixed Precision (AMP):**
- **Enabled**: True
- **Lý do**: Sử dụng FP16 cho một số operations, giảm memory usage và tăng tốc độ
- **Benefit**: Tăng tốc độ ~1.5-2x, giảm memory ~50%

**AMP Checks:**
- System tự động kiểm tra compatibility
- Passed ✅ trong training log

### 7.3. Hyperparameters

#### 7.3.1. Model Configuration

- **Model**: YOLOv8s (Small variant)
- **Parameters**: 11,166,560 (11.2M)
- **GFLOPs**: 28.8
- **Pretrained**: True (sử dụng pretrained weights từ COCO)

#### 7.3.2. Training Configuration

**Epochs:**
- **Value**: 120 epochs
- **Lý do**: Dataset balanced, cần nhiều epochs để convergence
- **Patience**: 0 (không early stopping, train đủ 120 epochs)

**Image Size:**
- **Value**: 640×640 pixels
- **Lý do**: Standard size cho YOLOv8, balance giữa accuracy và speed

**Batch Size:**
- **Value**: 28
- **Lý do**: Tối ưu cho P100 với YOLOv8s, fit trong VRAM
- **Effective Batch**: 64 (nbs=64, gradient accumulation)

#### 7.3.3. Optimizer Configuration

**Optimizer:**
- **Type**: SGD (Stochastic Gradient Descent)
- **Lý do**: SGD ổn định hơn AdamW cho balanced dataset
- **Momentum**: 0.937
- **Weight Decay**: 0.0005

**Learning Rate:**
- **Initial (lr0)**: 0.002
- **Final (lrf)**: 0.0001
- **Schedule**: Cosine annealing
- **Warmup**: 5 epochs
- **Warmup Momentum**: 0.8
- **Warmup Bias LR**: 0.1

**Lý do LR cao:**
- Balanced dataset → có thể học nhanh hơn
- Cosine schedule → giảm dần về cuối training

#### 7.3.4. Loss Functions

**Loss Weights:**
- **Box Loss**: 7.5 (localization)
- **Class Loss**: 0.5 (classification - giảm vì data balanced)
- **DFL Loss**: 1.5 (distribution focal loss)
- **Label Smoothing**: 0.0 (tắt vì data tốt)

**Lý do giảm Class Loss:**
- Dataset đã balanced → không cần weight cao cho classification
- Tập trung vào localization (Box Loss)

#### 7.3.5. Augmentation Strategy

**Augmentation Parameters (vừa phải vì data đã balanced):**

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| hsv_h | 0.015 | Hue adjustment |
| hsv_s | 0.7 | Saturation adjustment |
| hsv_v | 0.4 | Value (brightness) adjustment |
| degrees | 15.0 | Rotation (±15 degrees) |
| translate | 0.15 | Translation (15% of image size) |
| scale | 0.9 | Scale variation |
| shear | 5.0 | Shear transformation |
| perspective | 0.0005 | Perspective transformation |
| flipud | 0.0 | Vertical flip (disabled) |
| fliplr | 0.5 | Horizontal flip (50% probability) |
| mosaic | 1.0 | Mosaic augmentation (enabled) |
| mixup | 0.15 | Mixup augmentation (15% probability) |
| copy_paste | 0.1 | Copy-paste augmentation (10% probability) |

**Lý do augmentation vừa phải:**
- Dataset đã balanced → không cần augmentation mạnh
- Tránh over-augmentation → giảm noise

### 7.4. Training Process

#### 7.4.1. Training Speed

**Per Image Speed:**
- **Preprocess**: 0.6ms
- **Inference**: 3.8ms
- **Loss**: 0.0ms (negligible)
- **Postprocess**: 0.9ms
- **Total**: ~5.3ms per image

**Throughput:**
- Với batch size 28: ~5,283 images/second
- Một epoch (10,030 images): ~1.9 seconds
- 120 epochs: ~228 seconds (3.8 minutes) cho inference
- Tổng thời gian 8 giờ bao gồm: data loading, validation, logging, etc.

#### 7.4.2. Validation

**Validation Frequency:**
- **Every**: 1 epoch
- **Validation Set**: 9,217 images
- **Validation Time**: ~1 phút 13 giây per epoch (577 batches × 7.9 it/s)

**Metrics Tracked:**
- mAP50 (primary metric)
- mAP50-95
- Precision
- Recall
- Per-class metrics

#### 7.4.3. Model Saving

**Save Strategy:**
- **Save Period**: 10 epochs
- **Best Model**: Tự động save model có mAP50 cao nhất
- **Last Model**: Save model cuối cùng
- **Location**: `runs/detect/animal_balanced/weights/`

**Files Saved:**
- `best.pt`: Best model (highest mAP50)
- `last.pt`: Last epoch model
- `results.png`: Training curves
- `confusion_matrix.png`: Confusion matrix
- `val_batch*.jpg`: Validation examples

### 7.5. Training Curves và Monitoring

**Metrics được theo dõi:**

1. **Loss Curves:**
   - Train/Val Box Loss
   - Train/Val Class Loss
   - Train/Val DFL Loss
   - Total Loss

2. **mAP Curves:**
   - mAP50 (primary)
   - mAP50-95

3. **Precision/Recall/F1 Curves:**
   - Precision
   - Recall
   - F1-Score (harmonic mean của Precision và Recall)

**Phân tích Training:**
- Loss giảm dần và hội tụ
- mAP50 tăng dần trong quá trình training
- Validation metrics tương đối ổn định (không overfitting nghiêm trọng)

**Kết quả cuối cùng (Epoch 120):**
- mAP50: 0.6601
- mAP50-95: 0.3895
- Precision: 0.7283
- Recall: 0.5933
- F1-Score: 0.6540

---

## 8. KẾT QUẢ

### 8.1. Kết quả Training

#### 8.1.1. Overall Metrics

Sau 120 epochs training trên balanced dataset với YOLOv8s, mô hình đạt được các metrics sau trên validation set (9,217 images):

| Metric | Giá trị | Mô tả |
|--------|---------|-------|
| **mAP50** | 0.6601 (66.01%) | Mean Average Precision với IoU=0.5 |
| **mAP50-95** | 0.3895 (38.95%) | Mean Average Precision với IoU từ 0.5-0.95 |
| **Precision** | 0.7283 (72.83%) | Tỷ lệ detections đúng |
| **Recall** | 0.5933 (59.33%) | Tỷ lệ ground truth được detect |
| **F1-Score** | 0.6540 (65.40%) | Harmonic mean của Precision và Recall |

**Phân tích:**
- **Precision cao (72.83%)**: Model có độ tin cậy cao, ít false positives
- **Recall thấp (59.33%)**: Model conservative, bỏ sót một số đối tượng
- **F1-Score (65.40%)**: Phản ánh sự cân bằng giữa Precision và Recall, cho thấy model có performance trung bình tốt
- **Trade-off**: Precision-Recall trade-off cho thấy model ưu tiên accuracy hơn coverage

#### 8.1.2. So sánh với Baseline

| Metric | Baseline (Imbalanced) | Balanced (YOLOv8s) | Change |
|--------|----------------------|-------------------|--------|
| mAP50 | 0.6925 (69.25%) | 0.6601 (66.01%) | **-4.7%** |
| Precision | - | 0.7283 (72.83%) | - |
| Recall | - | 0.5933 (59.33%) | - |
| F1-Score | - | 0.6540 (65.40%) | - |

**Phân tích kết quả:**
- Kết quả **thấp hơn baseline** 4.7%, không đạt được mục tiêu cải thiện
- Có thể do các nguyên nhân:
  1. **Model quá lớn**: YOLOv8s (11.2M params) có thể quá lớn cho dataset 10k images → overfitting
  2. **Dataset size nhỏ**: 10k images có thể không đủ để model lớn học tốt
  3. **Hyperparameters**: Có thể cần điều chỉnh LR, augmentation, hoặc regularization

#### 8.1.3. Per-class Performance

**Top 10 Classes (mAP50 cao nhất):**

*Lưu ý: Cột "Images" là số ảnh trong validation set (9,217 ảnh) có chứa class tương ứng.*

| Class | mAP50 | Precision | Recall | Images (val) |
|-------|-------|-----------|--------|--------------|
| zebra | 0.945 | 0.881 | 0.896 | 143 |
| cat | 0.940 | 0.890 | 0.892 | 344 |
| giraffe | 0.936 | 0.922 | 0.892 | 207 |
| bear | 0.931 | 0.720 | 0.941 | 76 |
| elephant | 0.919 | 0.863 | 0.879 | 202 |
| train | 0.905 | 0.873 | 0.845 | 272 |
| fire hydrant | 0.910 | 0.887 | 0.862 | 153 |
| dog | 0.870 | 0.852 | 0.792 | 338 |
| stop sign | 0.872 | 0.800 | 0.834 | 169 |
| person | 0.779 | 0.832 | 0.674 | 5,013 |

**Bottom 10 Classes (mAP50 thấp nhất):**

| Class | mAP50 | Precision | Recall | Images (val) |
|-------|-------|-----------|--------|--------------|
| hair drier | 0.249 | 0.267 | 0.467 | 15 |
| book | 0.270 | 0.493 | 0.182 | 430 |
| carrot | 0.410 | 0.459 | 0.447 | 137 |
| apple | 0.437 | 0.498 | 0.424 | 124 |
| bench | 0.427 | 0.654 | 0.371 | 397 |
| toaster | 0.507 | 0.281 | 0.600 | 24 |
| handbag | 0.335 | 0.576 | 0.281 | 540 |
| backpack | 0.388 | 0.632 | 0.303 | 445 |
| knife | 0.360 | 0.650 | 0.258 | 330 |
| spoon | 0.373 | 0.681 | 0.249 | 283 |

**Nhận xét:**
- **Classes tốt**: Thường là động vật (zebra, cat, giraffe, bear, elephant) hoặc objects lớn, dễ nhận diện
- **Classes kém**: Thường là objects nhỏ (hair drier, toaster), hoặc objects có nhiều biến thể (book, handbag)
- **Person class**: mAP50 = 0.779, tốt nhưng không phải tốt nhất (do có nhiều biến thể)

### 8.2. Real-time Performance

#### 8.2.1. Inference Speed

**Per Image Speed:**
- **Preprocess**: 0.6ms
- **Inference**: 3.8ms
- **Postprocess**: 0.9ms
- **Total**: ~5.3ms per image

**Throughput:**
- Với batch size 1: ~189 FPS (lý thuyết)
- Với batch size 28: ~5,283 images/second

**Real-time Camera Performance:**
- **Frame Rate**: ~10-15 FPS (với frame skipping và image compression)
- **Latency**: ~100-200ms per frame (bao gồm network, processing, display)
- **Memory Usage**: ~2-3GB GPU memory

#### 8.2.2. Tracking Performance

**Track ID Stability:**
- Track IDs ổn định qua nhiều frames
- ID switching rate: Thấp (< 5% trong các test cases)
- Occlusion handling: Tốt, có thể track lại sau khi bị che khuất

**Association Accuracy:**
- IoU matching: Chính xác cho objects di chuyển chậm
- Feature matching: Hữu ích cho objects có appearance ổn định
- Combined cost: Weighted combination (0.5 IoU + 0.5 Feature) cho kết quả tốt

### 8.3. System Performance

#### 8.3.1. Web Application Performance

**Frontend:**
- **Load Time**: < 2 seconds
- **Responsiveness**: Smooth với 60 FPS UI
- **Memory**: ~100-200MB browser memory

**Backend:**
- **API Response Time**: 
  - `/api/detect`: ~100-200ms (single image)
  - `/api/detect-video`: ~150-250ms (frame với tracking)
- **Concurrent Requests**: Hỗ trợ nhiều sessions đồng thời
- **Session Management**: Auto-cleanup sau 5 phút không hoạt động

#### 8.3.2. Resource Usage

**GPU:**
- **Training**: ~15-16GB VRAM (với batch size 28)
- **Inference**: ~2-3GB VRAM
- **Utilization**: ~80-90% trong training, ~40-60% trong inference

**CPU:**
- **Data Loading**: 20 workers, ~50-70% CPU usage
- **Inference**: Single-threaded, ~10-20% CPU usage

**Memory:**
- **Training**: ~8-10GB RAM
- **Inference**: ~2-4GB RAM

### 8.4. Error Analysis

#### 8.4.1. Common Errors

**False Positives:**
- Objects tương tự nhau (ví dụ: handbag vs backpack)
- Background patterns bị nhầm là objects
- Small objects với low confidence

**False Negatives:**
- Objects quá nhỏ (< 32×32 pixels)
- Objects bị che khuất nhiều
- Objects ở góc ảnh hoặc ngoài vùng quan tâm
- Objects với appearance không quen thuộc

**Localization Errors:**
- Bounding boxes không chính xác cho objects có shape phức tạp
- Multiple objects gần nhau bị merge thành một box

#### 8.4.2. Per-class Error Patterns

**Classes có nhiều False Positives:**
- book (Precision = 0.493): Dễ nhầm với các objects phẳng khác
- handbag (Precision = 0.576): Tương tự backpack, suitcase
- knife (Precision = 0.650): Nhỏ, khó phân biệt

**Classes có nhiều False Negatives:**
- book (Recall = 0.182): Rất thấp, có thể do quá nhiều biến thể
- spoon (Recall = 0.249): Nhỏ, khó detect
- handbag (Recall = 0.281): Tương tự

---

## 9. DEMO / ỨNG DỤNG

### 9.1. Giao diện Hệ thống

#### 9.1.1. Home Screen

Màn hình chính cung cấp 2 lựa chọn:

**📷 Camera Mode:**
- Nhận diện đối tượng real-time từ webcam
- Hiển thị bounding boxes với track IDs
- Audio feedback cho đối tượng mới
- FPS counter và detection rate

**🖼️ Image Mode:**
- Upload ảnh tĩnh để nhận diện
- Drag & drop interface
- Hiển thị kết quả với bounding boxes
- Bảng kết quả chi tiết

#### 9.1.2. Camera Mode Interface

**Layout:**
- **Left Panel (2/3 width)**: Video feed với bounding boxes overlay
- **Right Panel (1/3 width)**: Results table với active tracks
- **Bottom Bar**: Status indicators và audio controls

**Features:**
- Real-time detection với frame rate ~10-15 FPS
- Track IDs hiển thị trên bounding boxes
- Color coding:
  - Green: New tracks
  - Blue: High confidence (>0.7)
  - Yellow: Medium confidence (0.5-0.7)
  - Red: Low confidence (<0.5)
- Detection indicator khi đang xử lý
- Object count badge

#### 9.1.3. Image Mode Interface

**Layout:**
- **Left Panel (2/3 width)**: Image preview với bounding boxes
- **Right Panel (1/3 width)**: Detect button, audio controls, results table

**Features:**
- Image upload với drag & drop
- Zoom & pan để xem chi tiết
- Results table sortable theo confidence/class
- Statistics: total objects, classes, avg confidence
- Audio feedback với gom theo lớp

### 9.2. Tính năng Chính

#### 9.2.1. Real-time Camera Detection

**Workflow:**
1. User chọn "Camera" mode
2. Hệ thống yêu cầu quyền truy cập camera
3. Camera stream bắt đầu
4. Mỗi frame được capture và gửi đến backend
5. Backend xử lý với YOLO + DeepSORT
6. Kết quả trả về với track IDs
7. Frontend hiển thị bounding boxes và update results table

**Optimizations:**
- Frame skipping: Skip nếu đang detect
- Image compression: Resize 320×240, quality 0.6
- Request queue: Chỉ 1 request active
- AbortController: Cancel requests cũ

#### 9.2.2. Image Upload và Detection

**Workflow:**
1. User upload ảnh (drag & drop hoặc click)
2. File validation (type, size)
3. Preview ảnh
4. User click "Detect Objects"
5. Backend xử lý với YOLO
6. Kết quả hiển thị với bounding boxes
7. Results table và statistics
8. Audio feedback (nếu enabled)

**Supported Formats:**
- JPG, JPEG
- PNG
- BMP
- WEBP
- TIFF

**Max File Size:** 10MB

#### 9.2.3. Multi-Object Tracking

**Features:**
- Stable track IDs qua nhiều frames
- Track history (last 30 states)
- Occlusion handling
- ID switching prevention

**Visualization:**
- Track ID hiển thị trên bounding box
- Color coding theo confidence và is_new
- "NEW" label cho tracks mới

#### 9.2.4. Audio Feedback

**Text-to-Speech (TTS):**
- Sử dụng Web Speech API
- Language: Vietnamese (vi-VN)
- Rate: 1.0 (normal speed)
- Volume: 1.0 (maximum)

**Gom kết quả theo lớp:**
- Thay vì: "Phát hiện xe tải. Phát hiện xe tải. Phát hiện người."
- Gom thành: "Phát hiện 2 xe tải. Phát hiện 1 người."

**Camera Mode:**
- Chỉ đọc đối tượng mới (dựa trên track IDs)
- Tránh spam audio khi đối tượng vẫn còn trong frame
- Debounce mechanism để tránh đọc quá nhiều

**Controls:**
- Bật/tắt audio
- Đọc lại kết quả gần nhất
- Auto-stop khi chuyển mode

### 9.3. Use Cases

#### 9.3.1. Surveillance và An ninh

**Ứng dụng:**
- Giám sát khu vực công cộng
- Phát hiện đối tượng đáng ngờ
- Đếm số lượng người/vehicles
- Tracking đối tượng qua nhiều cameras

**Tính năng hữu ích:**
- Real-time detection
- Multi-object tracking
- Statistics và reporting

#### 9.3.2. Accessibility - Hỗ trợ Người khiếm thị

**Ứng dụng:**
- Mô tả môi trường xung quanh bằng audio
- Phát hiện obstacles và objects
- Hướng dẫn navigation

**Tính năng hữu ích:**
- Audio feedback bằng tiếng Việt
- Gom kết quả theo lớp (dễ hiểu)
- Chỉ đọc đối tượng mới (tránh spam)

#### 9.3.3. Education và Học tập

**Ứng dụng:**
- Học về object detection
- Demo các mô hình AI
- Thực hành với real-world data

**Tính năng hữu ích:**
- Visual feedback với bounding boxes
- Statistics và metrics
- Interactive interface

### 9.4. Screenshots và Demo

**Các màn hình chính:**
1. Home screen với 2 options
2. Camera mode với video feed và results
3. Image mode với uploaded image và results
4. Results table với sortable columns
5. Audio controls và settings

**Demo scenarios:**
- Real-time detection trong phòng
- Upload ảnh street scene
- Tracking multiple people
- Audio feedback demonstration

---

## 10. ĐÁNH GIÁ & THẢO LUẬN

### 10.1. Đánh giá Kết quả

#### 10.1.1. Thành công

**1. Pipeline xử lý dữ liệu hiệu quả:**
- Smart sampling algorithm thành công trong việc chọn ảnh chất lượng cao
- Dataset được balanced từ 321:1 xuống ~1:1
- Augmentation strategy phù hợp cho classes thiếu
- Tổng thời gian xử lý: ~20 phút cho 82k images

**2. Hệ thống web application hoàn chỉnh:**
- Real-time camera detection hoạt động ổn định
- Multi-object tracking với DeepSORT duy trì track IDs tốt
- Audio feedback hữu ích cho người khiếm thị
- UI/UX responsive và user-friendly

**3. Tối ưu hóa performance:**
- Frame skipping và image compression giảm load
- Request queue tránh race conditions
- Session management tự động cleanup

**4. Precision và F1-Score tốt:**
- Precision = 72.83% cho thấy model có độ tin cậy cao
- F1-Score = 65.40% phản ánh sự cân bằng tốt giữa Precision và Recall
- Ít false positives, phù hợp cho applications cần accuracy

#### 10.1.2. Hạn chế

**1. mAP50 thấp hơn baseline:**
- mAP50 = 66.01% vs baseline 69.25% (-4.7%)
- Không đạt được mục tiêu 0.78-0.83
- Cần phân tích và cải thiện

**2. Recall thấp:**
- Recall = 59.33% cho thấy model bỏ sót nhiều đối tượng
- Model quá conservative
- Cần điều chỉnh confidence threshold hoặc loss weights

**3. Một số classes có performance kém:**
- hair drier: mAP50 = 0.249 (rất thấp)
- book: mAP50 = 0.270 (rất thấp)
- toaster: mAP50 = 0.507 (thấp)
- Có thể do ít dữ liệu hoặc objects quá nhỏ

**4. Model size có thể không phù hợp:**
- YOLOv8s (11.2M params) có thể quá lớn cho dataset 10k images
- Có thể dẫn đến overfitting tiềm ẩn

### 10.2. Phân tích Nguyên nhân

#### 10.2.1. Tại sao mAP50 thấp hơn baseline?

**Nguyên nhân có thể:**

1. **Model quá lớn so với dataset size:**
   - YOLOv8s (11.2M params) vs dataset 10k images
   - Rule of thumb: Cần ~100-1000 samples per parameter
   - 10k images có thể không đủ cho 11.2M params
   - **Giải pháp**: Thử YOLOv8n (3.2M params) phù hợp hơn

2. **Dataset size nhỏ:**
   - Giảm từ 82k xuống 10k (87.8% reduction)
   - Mất nhiều diversity và variation
   - **Giải pháp**: Tăng dataset size hoặc augmentation mạnh hơn

3. **Hyperparameters chưa tối ưu:**
   - Learning rate có thể cần điều chỉnh
   - Augmentation có thể cần mạnh hơn
   - Regularization có thể cần tăng
   - **Giải pháp**: Hyperparameter tuning

4. **Loss weights:**
   - Class loss = 0.5 (thấp) có thể không đủ
   - Box loss = 7.5 (cao) có thể quá tập trung vào localization
   - **Giải pháp**: Điều chỉnh loss weights

#### 10.2.2. Tại sao Recall thấp?

**Nguyên nhân:**

1. **Model conservative:**
   - Precision cao (72.83%) nhưng Recall thấp (59.33%)
   - Model ưu tiên accuracy hơn coverage
   - Có thể do confidence threshold cao

2. **Small objects:**
   - Nhiều objects nhỏ bị bỏ sót
   - YOLOv8 có thể kém với objects < 32×32 pixels

3. **Occlusion:**
   - Objects bị che khuất khó detect
   - Cần data augmentation với occlusion

**Giải pháp:**
- Giảm confidence threshold
- Tăng class loss weight
- Augmentation với small objects và occlusion

#### 10.2.3. Tại sao một số classes kém?

**Classes kém (mAP50 < 0.5):**
- hair drier (0.249): Chỉ có 15 images → quá ít dữ liệu
- book (0.270): Quá nhiều biến thể, khó học
- carrot (0.410): Nhỏ, khó detect
- apple (0.437): Tương tự carrot

**Nguyên nhân:**
1. **Ít dữ liệu**: hair drier chỉ có 15 images (sau augmentation)
2. **High variation**: book có nhiều loại, khó generalize
3. **Small size**: carrot, apple thường nhỏ trong ảnh
4. **Similar appearance**: Dễ nhầm với objects tương tự

**Giải pháp:**
- Tăng dữ liệu cho classes thiếu
- Augmentation mạnh hơn
- Focal loss để tập trung vào hard examples

### 10.3. So sánh với Các Phương pháp Khác

#### 10.3.1. Baseline Comparison

| Method | Dataset | mAP50 | Notes |
|--------|---------|-------|-------|
| Baseline | Imbalanced (82k) | 0.6925 | Full dataset, imbalanced |
| Our Method | Balanced (10k) | 0.6601 | Balanced, YOLOv8s |
| Difference | - | **-4.7%** | Lower than baseline |

**Phân tích:**
- Balanced dataset không cải thiện kết quả như kỳ vọng
- Có thể do dataset size nhỏ hoặc model không phù hợp
- Cần thử YOLOv8n hoặc tăng dataset size

#### 10.3.2. Model Size Comparison

**Đề xuất so sánh (chưa thực hiện):**

| Model | Parameters | Dataset Size | Expected mAP50 |
|-------|-----------|--------------|----------------|
| YOLOv8n | 3.2M | 10k | 0.70-0.75 (ước tính) |
| YOLOv8s | 11.2M | 10k | 0.6601 (thực tế) |
| YOLOv8s | 11.2M | 20k+ | 0.75+ (ước tính) |

**Kết luận:**
- YOLOv8n có thể phù hợp hơn cho dataset 10k images
- YOLOv8s cần dataset lớn hơn để phát huy capacity

### 10.4. Lessons Learned

#### 10.4.1. Model Size vs Dataset Size

**Lesson:**
- Model size phải phù hợp với dataset size
- Rule of thumb: ~100-1000 samples per parameter
- YOLOv8s (11.2M params) cần ~1-10M samples để optimal
- Dataset 10k images → nên dùng YOLOv8n (3.2M params)

**Application:**
- Khi chọn model, cần xem xét dataset size
- Không phải model lớn hơn luôn tốt hơn
- Cần balance giữa capacity và overfitting risk

#### 10.4.2. Balanced Dataset không đảm bảo cải thiện

**Lesson:**
- Balanced dataset là cần thiết nhưng không đủ
- Cần kết hợp với model size phù hợp
- Cần hyperparameters tối ưu
- Cần đủ dữ liệu cho mỗi class

**Application:**
- Balanced dataset là bước đầu, không phải giải pháp cuối cùng
- Cần xem xét nhiều yếu tố: model, hyperparameters, augmentation

#### 10.4.3. Precision vs Recall Trade-off

**Lesson:**
- Model có thể ưu tiên Precision hoặc Recall
- Cần điều chỉnh loss weights và thresholds
- Tùy application mà chọn trade-off phù hợp

**Application:**
- Surveillance: Cần Recall cao (không bỏ sót)
- Medical: Cần Precision cao (ít false positives)
- General: Cần balance cả hai

#### 10.4.4. Quality Score Algorithm

**Lesson:**
- Smart sampling với quality score hiệu quả
- Chọn ảnh tốt quan trọng hơn số lượng
- Quality > Quantity trong một số trường hợp

**Application:**
- Khi có dataset lớn, nên chọn subset chất lượng cao
- Quality score có thể customize cho từng application

---

## 11. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 11.1. Kết luận

Nghiên cứu này đã xây dựng thành công một hệ thống nhận diện đối tượng hoàn chỉnh với các thành phần chính:

**1. Pipeline xử lý dữ liệu thông minh:**
- Phát triển thuật toán smart sampling dựa trên quality score
- Cân bằng dataset từ 321:1 xuống ~1:1
- Giảm dataset từ 82k xuống 10k images nhưng vẫn giữ chất lượng

**2. Training mô hình YOLOv8s:**
- Tối ưu hóa cho GPU Tesla P100
- Training 120 epochs trong 8 giờ
- Đạt được mAP50 = 66.01%, Precision = 72.83%, Recall = 59.33%, F1-Score = 65.40%

**3. Hệ thống web application:**
- Real-time camera detection với tracking
- Image upload và detection
- Audio feedback bằng tiếng Việt
- UI/UX responsive và user-friendly

**4. Phân tích và đánh giá:**
- So sánh với baseline và phân tích nguyên nhân
- Đưa ra lessons learned và đề xuất cải thiện

Mặc dù kết quả mAP50 thấp hơn baseline (-4.7%), nghiên cứu đã cung cấp insights quan trọng về mối quan hệ giữa model size và dataset size, gợi ý rằng YOLOv8n có thể phù hợp hơn cho dataset 10k images.

### 11.2. Đóng góp

**1. Smart Sampling Algorithm:**
- Quality score dựa trên số classes, bbox area, và vị trí
- Hiệu quả trong việc chọn ảnh chất lượng cao
- Có thể áp dụng cho các dataset khác

**2. Tối ưu hóa Training cho P100:**
- Auto-detection GPU và tối ưu batch size
- Mixed Precision Training (AMP)
- Workers và cache optimization

**3. Hệ thống Application hoàn chỉnh:**
- Real-time tracking với DeepSORT
- Audio feedback cho accessibility
- Performance optimization (frame skipping, compression)

**4. Phân tích và Insights:**
- Mối quan hệ model size vs dataset size
- Precision-Recall trade-off analysis
- Per-class performance analysis

### 11.3. Hạn chế

**1. Kết quả không đạt mục tiêu:**
- mAP50 = 66.01% < mục tiêu 78-83%
- Thấp hơn baseline 4.7%

**2. Model size có thể không phù hợp:**
- YOLOv8s (11.2M params) có thể quá lớn cho 10k images
- Chưa thử YOLOv8n để so sánh

**3. Dataset size nhỏ:**
- Giảm 87.8% từ 82k xuống 10k
- Có thể mất diversity

**4. Một số classes performance kém:**
- hair drier, book, carrot có mAP50 rất thấp
- Cần thêm dữ liệu hoặc augmentation

### 11.4. Hướng phát triển

#### 11.4.1. Model và Training

**1. Thử YOLOv8n:**
- YOLOv8n (3.2M params) phù hợp hơn với dataset 10k images
- So sánh kết quả với YOLOv8s
- Có thể đạt mAP50 cao hơn

**2. Hyperparameter Tuning:**
- Learning rate scheduling
- Augmentation strategy
- Loss weights optimization
- Regularization techniques

**3. Data Augmentation mạnh hơn:**
- Tăng effective dataset size
- Mixup, CutMix, Mosaic
- Synthetic data generation

**4. Transfer Learning:**
- Fine-tuning từ pretrained model
- Domain adaptation
- Multi-task learning

#### 11.4.2. Dataset

**1. Tăng Dataset Size:**
- Thêm dữ liệu từ các nguồn khác
- Data augmentation mạnh hơn
- Synthetic data generation

**2. Cải thiện Classes thiếu:**
- Thu thập thêm dữ liệu cho hair drier, toaster
- Augmentation tập trung cho classes kém
- Active learning để chọn ảnh quan trọng

**3. Dataset Diversity:**
- Thêm các scenarios khác nhau
- Điều kiện ánh sáng, góc chụp đa dạng
- Background và context đa dạng

#### 11.4.3. Tracking

**1. Nâng cấp Feature Extraction:**
- Thay histogram-based bằng CNN-based features
- Sử dụng ReID (Re-identification) models
- ResNet hoặc MobileNet cho feature extraction

**2. Cải thiện Association:**
- Adaptive cost matrix weights
- Temporal consistency
- Appearance và motion models

**3. Occlusion Handling:**
- Better prediction khi bị che khuất
- Long-term tracking
- Re-identification sau occlusion

#### 11.4.4. Application

**1. Mobile Application:**
- iOS và Android apps
- Edge deployment với TensorFlow Lite
- On-device inference

**2. Video Processing:**
- Batch video processing
- Video analysis và reporting
- Export results to video

**3. Advanced Features:**
- Object counting
- Behavior analysis
- Alert system
- Multi-camera support

**4. Performance Optimization:**
- Model quantization
- Pruning
- Knowledge distillation
- Faster inference

#### 11.4.5. Accessibility

**1. Cải thiện Audio Feedback:**
- Natural language generation
- Context-aware descriptions
- Priority-based announcements

**2. Multi-language Support:**
- Hỗ trợ nhiều ngôn ngữ
- Language detection
- Customizable translations

**3. Haptic Feedback:**
- Vibration patterns
- Spatial audio
- Multi-modal feedback

---

## 12. TÀI LIỆU THAM KHẢO

### 12.1. Papers và Research

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). "You Only Look Once: Unified, Real-Time Object Detection." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

2. Redmon, J., & Farhadi, A. (2017). "YOLO9000: Better, Faster, Stronger." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

3. Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement." *arXiv preprint arXiv:1804.02767*.

4. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). "YOLOv4: Optimal Speed and Accuracy of Object Detection." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

5. Ultralytics. (2023). "YOLOv8 Documentation." *Ultralytics*. https://docs.ultralytics.com/

6. Wojke, N., Bewley, A., & Paulus, D. (2017). "Simple Online and Realtime Tracking with a Deep Association Metric." *Proceedings of the IEEE International Conference on Image Processing (ICIP)*.

7. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). "Simple Online and Realtime Tracking." *Proceedings of the IEEE International Conference on Image Processing (ICIP)*.

8. Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014). "Microsoft COCO: Common Objects in Context." *European Conference on Computer Vision (ECCV)*.

### 12.2. Documentation và Frameworks

9. Ultralytics. (2023). "Ultralytics YOLOv8." *GitHub*. https://github.com/ultralytics/ultralytics

10. FastAPI. (2023). "FastAPI Documentation." *FastAPI*. https://fastapi.tiangolo.com/

11. React. (2023). "React Documentation." *React*. https://react.dev/

12. Tailwind CSS. (2023). "Tailwind CSS Documentation." *Tailwind CSS*. https://tailwindcss.com/

13. PyTorch. (2023). "PyTorch Documentation." *PyTorch*. https://pytorch.org/docs/

14. OpenCV. (2023). "OpenCV Documentation." *OpenCV*. https://docs.opencv.org/

### 12.3. Datasets

15. Lin, T. Y., et al. (2014). "COCO Dataset." *Common Objects in Context*. https://cocodataset.org/

16. COCO 2014 Dataset for YOLOv3. (2023). *Kaggle*. https://www.kaggle.com/datasets/

### 12.4. Tools và Libraries

17. Scipy. (2023). "Scipy Documentation." *Scipy*. https://docs.scipy.org/

18. FilterPy. (2023). "FilterPy Documentation." *FilterPy*. https://filterpy.readthedocs.io/

19. Axios. (2023). "Axios Documentation." *Axios*. https://axios-http.com/

20. Web Speech API. (2023). "MDN Web Docs." *Mozilla Developer Network*. https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API

---

**Kết thúc báo cáo**

---

