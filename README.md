# 🎯 Object Detection System

Hệ thống nhận diện đối tượng sử dụng YOLOv8 với giao diện web React và backend FastAPI.

## 📋 Mô Tả

Ứng dụng web cho phép người dùng upload ảnh hoặc sử dụng camera để nhận diện đối tượng sử dụng mô hình YOLOv8n đã được training. Hệ thống hỗ trợ 80 lớp đối tượng khác nhau, hiển thị kết quả với bounding boxes, thống kê chi tiết và cho phép tùy chỉnh các tham số detection. Hệ thống còn hỗ trợ Text-to-Speech (TTS) bằng tiếng Việt để hỗ trợ người dùng khiếm thị.

**Kết quả:**
- mAP50: **0.7565** (75.65%)
- Precision: **0.7140**
- Recall: **0.7469**
- Cải thiện **+9.2%** so với baseline

## 🏗️ Cấu Trúc Dự Án

```
Animal-Detection-System-Data-Mining-Project/  # Note: Tên folder (có thể giữ nguyên)
├── backend/                      # FastAPI backend
│   ├── app.py                    # Main API application
│   ├── inference.py              # ObjectDetector class
│   └── requirements.txt          # Python dependencies
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── services/             # API service
│   │   └── App.jsx               # Main app component
│   └── package.json              # Node dependencies
├── code_train_model/             # Training scripts
│   ├── data_preparation_pro.py   # Data preparation pipeline
│   └── model_training_optimized.py
├── best.pt                       # Trained YOLOv8n model
├── doc/                          # Documentation
│   └── BAO_CAO.md                # Báo cáo đồ án
├── start_backend.sh              # Script chạy backend
└── start_frontend.sh             # Script chạy frontend
```

## 🚀 Cài Đặt và Chạy

### Yêu Cầu Hệ Thống

- **Python**: 3.8+
- **Node.js**: 14+ (khuyến nghị 16+)
- **Model file**: `best.pt` (đã có sẵn)

### Bước 1: Cài Đặt Dependencies

**Option A: Quick Install (CPU-only, NHANH - Khuyến nghị cho test)**
```bash
chmod +x quick_install.sh
./quick_install.sh
```
⏱️ Thời gian: 2-5 phút | 📦 Download: ~200MB

**Option B: Full GPU Install (Nếu có GPU NVIDIA)**
```bash
chmod +x install_gpu.sh
./install_gpu.sh
```
⏱️ Thời gian: 15-30 phút | 📦 Download: ~3GB

### Bước 2: Chạy Hệ Thống

**Terminal 1 - Backend:**
```bash
chmod +x start_backend.sh
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
chmod +x start_frontend.sh
./start_frontend.sh
```

### Bước 3: Test Hệ Thống

Xem hướng dẫn test chi tiết ở phần **"Hướng Dẫn Test"** bên dưới.

### Cách 2: Chạy Thủ Công

#### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend chạy tại: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

#### Frontend

```bash
cd frontend
npm install
npm start
```

Frontend tự động mở tại: `http://localhost:3000`

## 📖 Hướng Dẫn Sử Dụng

### 1. Chọn Chế Độ
Khi khởi động ứng dụng, bạn sẽ thấy màn hình Home với 2 lựa chọn:
- **📷 Camera**: Nhận diện đối tượng real-time từ camera
- **🖼️ Hình Ảnh**: Upload và nhận diện ảnh tĩnh

### 2. Chế Độ Camera
- Click vào "📷 Camera" để bắt đầu
- Cho phép truy cập camera khi được yêu cầu
- Hệ thống sẽ tự động nhận diện đối tượng và hiển thị bounding boxes
- Có thể điều chỉnh "Khoảng thời gian nhận diện" (300ms - 2000ms)
- Audio feedback sẽ thông báo các đối tượng được phát hiện bằng tiếng Việt

### 3. Chế Độ Hình Ảnh
- Click vào "🖼️ Hình Ảnh" để upload ảnh
- **Upload**: Click "Chọn Ảnh" hoặc drag & drop
- **Điều Chỉnh Settings**:
  - **Confidence Threshold** (0.0 - 1.0): Mặc định 0.25
  - **IoU Threshold** (0.0 - 1.0): Mặc định 0.45
- **Nhận Diện**: Click "Nhận Diện" để bắt đầu
- **Kết quả hiển thị**:
  - Ảnh với bounding boxes
  - Bảng detections chi tiết (có thể sắp xếp)
  - Thống kê tổng hợp
  - Audio feedback bằng tiếng Việt

### 4. Tính Năng Khác
- **Audio Feedback**: Hệ thống tự động phát âm kết quả bằng tiếng Việt
- **Zoom & Pan**: Phóng to và kéo thả ảnh để xem chi tiết
- **Sorting**: Sắp xếp kết quả theo confidence hoặc tên đối tượng

## 🎯 Tính Năng

### Core Features
- ✅ **Real-time Camera Detection**: Nhận diện đối tượng từ camera với bounding boxes
- ✅ **Image Upload**: Upload và nhận diện ảnh tĩnh (drag & drop)
- ✅ **80 Classes Detection**: Nhận diện 80 lớp đối tượng khác nhau với YOLOv8
- ✅ **Bounding Boxes**: Hiển thị khung bao quanh đối tượng được phát hiện
- ✅ **Results Table**: Bảng kết quả chi tiết với khả năng sắp xếp
- ✅ **Statistics**: Thống kê tổng hợp (phân bố classes, confidence)

### Advanced Features
- ✅ **Text-to-Speech (TTS)**: Audio feedback bằng tiếng Việt
- ✅ **Customizable Thresholds**: Tùy chỉnh confidence và IoU thresholds
- ✅ **Image Zoom & Pan**: Phóng to và kéo thả để xem chi tiết
- ✅ **Localization**: Giao diện và kết quả hoàn toàn bằng tiếng Việt
- ✅ **Responsive UI**: Giao diện responsive, tối ưu cho mọi thiết bị
- ✅ **Accessibility**: Hỗ trợ người dùng khiếm thị với audio feedback

## 🔧 API Endpoints

### `GET /api/model-info`
Lấy thông tin model (số classes, danh sách classes, thresholds mặc định)

### `POST /api/detect`
Nhận diện đối tượng trong 1 ảnh

**Request:**
- `file`: File ảnh (multipart/form-data)
- `conf_threshold`: float (optional, default: 0.25)
- `iou_threshold`: float (optional, default: 0.45)

**Response:**
```json
{
  "success": true,
  "detections": [...],
  "image_base64": "data:image/jpeg;base64,...",
  "statistics": {...}
}
```

### `POST /api/compare-thresholds`
So sánh kết quả với các confidence threshold khác nhau

## 📊 Model Performance

### Metrics

| Metric | Giá trị |
|--------|---------|
| mAP50 | 0.7565 (75.65%) |
| mAP50-95 | 0.6322 (63.22%) |
| Precision | 0.7140 |
| Recall | 0.7469 |
| F1-Score | 0.7301 |

### Training Details

- **Model**: YOLOv8n (nano)
- **Dataset**: 28,184 samples (80 classes)
- **Train/Val**: 22,518 / 5,666 (80/20)
- **Epochs**: 100
- **Training time**: 8 giờ 21 phút
- **Hardware**: Tesla P100 GPU (16GB)

### Improvement

- **Baseline** (imbalanced data): mAP50 = 0.6925
- **After balancing**: mAP50 = 0.7565
- **Improvement**: **+9.2%** 🎉

## 🧪 Hướng Dẫn Test (Tóm tắt)

### 1. Cài Đặt Nhanh Cho Test

```bash
./quick_install.sh
```

Hoặc nếu có GPU:

```bash
./install_gpu.sh
```

### 2. Chạy Backend & Frontend

```bash
# Terminal 1 - Backend
./start_backend.sh

# Terminal 2 - Frontend
./start_frontend.sh
```

Mở `http://localhost:3000` để sử dụng.

### 3. Test Nhanh

- **Image mode**: Chọn "Hình Ảnh" → upload ảnh → "Nhận Diện Đối Tượng" → kiểm tra bounding box, bảng kết quả và audio (gom theo lớp, ví dụ: "Phát hiện 2 xe tải. Phát hiện 1 người").
- **Camera mode**: Chọn "Camera" → cho phép quyền camera → kiểm tra Track IDs ổn định, audio chỉ đọc đối tượng mới (theo `track_id`), không lặp lại đối tượng cũ.

---

## 📚 Tài Liệu

- **Báo cáo**: Xem file `doc/BAO_CAO.md` để biết chi tiết về dự án

## 🐛 Troubleshooting

### Backend không chạy được
1. Kiểm tra Python version: `python3 --version` (cần 3.8+)
2. Kiểm tra model path trong `backend/app.py`
3. Kiểm tra dependencies: `pip install -r backend/requirements.txt`

### Frontend không kết nối được backend
1. Đảm bảo backend đang chạy tại `http://localhost:8000`
2. Kiểm tra CORS settings trong `backend/app.py`
3. Kiểm tra API URL trong `frontend/src/services/api.js`

### Model không load được
1. Kiểm tra file `best.pt` có tồn tại trong thư mục gốc
2. Kiểm tra đường dẫn `MODEL_PATH` trong `backend/app.py`

## 📝 Ghi Chú

- File upload được lưu tạm trong system temp directory và tự động xóa sau khi xử lý
- Model được load một lần khi khởi động backend
- Frontend sử dụng Tailwind CSS cho styling
- Audio feedback sử dụng Web Speech API (SpeechSynthesis)
- Camera detection sử dụng MediaDevices API với tối ưu hóa performance
- Hệ thống hỗ trợ cả desktop và mobile browsers

## 🔒 Security Features

- ✅ Path traversal protection (sanitized filenames)
- ✅ File size validation (max 10MB)
- ✅ CORS configuration (configurable via environment variable)
- ✅ Request timeout (30s for detection, 60s for batch)
- ✅ Input validation (thresholds, file types)

## ⚡ Performance Optimizations

- ✅ Request queue với AbortController (tránh race conditions)
- ✅ Frame skipping logic (giảm server load)
- ✅ Image optimization (resize 320x240, quality 0.6)
- ✅ Efficient bounding box scaling
- ✅ Memory leak prevention

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

---

**Sinh viên:** Phan Văn Tài - MSSV: 2202081  
**Giảng viên hướng dẫn:** Tiến sĩ Trần Ngọc Anh  
**Trường Đại học Tân Tạo - Khoa Công nghệ Thông tin**
# final_project_datamining
