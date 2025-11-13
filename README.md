# 🐾 Animal Detection System

Hệ thống nhận diện động vật sử dụng YOLO với giao diện web React và backend FastAPI.

## 📋 Mô Tả

Ứng dụng web cho phép người dùng upload ảnh và nhận diện động vật trong ảnh sử dụng mô hình YOLO đã được training. Hệ thống hiển thị kết quả với bounding boxes, thống kê chi tiết và cho phép tùy chỉnh các tham số detection.

## 🏗️ Kiến Trúc

```
the_end/
├── best.pt                    # YOLO model file
├── backend/                   # FastAPI backend
│   ├── app.py                 # Main API application
│   ├── inference.py           # AnimalDetector class
│   ├── requirements.txt       # Python dependencies
│   └── uploads/              # Temporary upload folder
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── services/          # API service
│   │   └── App.jsx            # Main app component
│   └── package.json           # Node dependencies
├── start_backend.sh           # Script chạy backend
└── start_frontend.sh          # Script chạy frontend
```

## 🚀 Cài Đặt và Chạy

### Yêu Cầu Hệ Thống

- **Python**: 3.8+
- **Node.js**: 14+ (khuyến nghị 16+)
- **Model file**: `best.pt` (đã có sẵn)

### Cách 1: Sử dụng Scripts (Khuyến nghị)

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

### Cách 2: Chạy Thủ Công

#### Bước 1: Cài Đặt Backend

```bash
cd backend

# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

#### Bước 2: Chạy Backend

```bash
cd backend
python app.py
```

Backend chạy tại: `http://localhost:8000`  
API docs: `http://localhost:8000/docs`

#### Bước 3: Cài Đặt Frontend

```bash
cd frontend
npm install
```

**Lưu ý:** Nếu chưa có npm:
```bash
# Ubuntu/Debian
sudo apt install npm

# Hoặc download từ: https://nodejs.org/
```

#### Bước 4: Chạy Frontend

```bash
cd frontend
npm start
```

Frontend tự động mở tại: `http://localhost:3000`

## 📖 Hướng Dẫn Sử Dụng

### 1. Upload Ảnh

- **Chọn 1 ảnh**: Click "Select Single Image" hoặc kéo thả ảnh vào vùng upload
- **Chọn nhiều ảnh**: Click "Select Multiple Images" để xử lý batch

### 2. Điều Chỉnh Settings

- **Confidence Threshold** (0.0 - 1.0):
  - **Low (0.1)**: Nhiều detections, có thể có false positives
  - **Medium (0.25)**: Cân bằng (mặc định)
  - **High (0.5+)**: Chỉ detections chắc chắn

- **IoU Threshold**: Ngưỡng IoU cho Non-Maximum Suppression (mặc định 0.45)

### 3. Nhận Diện

- Click "Detect" để bắt đầu detection
- Kết quả hiển thị:
  - Ảnh có bounding boxes (tab "Result")
  - Bảng chi tiết các detections (sortable)
  - Thống kê tổng hợp (phân bố classes, confidence range)

### 4. So Sánh Thresholds

- Click "Compare Thresholds" để xem kết quả với nhiều threshold khác nhau
- Giúp tìm threshold tối ưu cho ảnh của bạn

### 5. Batch Processing

- Khi chọn nhiều ảnh, bấm "Detect" một lần để xử lý tất cả
- Sử dụng nút Previous/Next hoặc phím mũi tên (← →) để chuyển giữa các ảnh
- Kết quả đã detect sẽ tự động hiển thị khi chuyển ảnh

## 🎯 Tính Năng

- ✅ Upload và preview ảnh (drag & drop)
- ✅ Nhận diện động vật với YOLO
- ✅ Hiển thị bounding boxes trên ảnh
- ✅ Bảng kết quả chi tiết (sortable)
- ✅ Thống kê tổng hợp (phân bố classes, confidence)
- ✅ Tùy chỉnh confidence và IoU thresholds
- ✅ So sánh kết quả với nhiều thresholds
- ✅ Batch processing (nhiều ảnh cùng lúc)
- ✅ Keyboard shortcuts (arrow keys)
- ✅ File validation (format, size)
- ✅ Giao diện responsive, dễ sử dụng

## 🔧 API Endpoints

### `GET /api/model-info`
Lấy thông tin model (số classes, danh sách classes, thresholds mặc định)

### `POST /api/detect`
Nhận diện động vật trong 1 ảnh

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

### `POST /api/detect-batch`
Nhận diện nhiều ảnh cùng lúc (tối đa 20 ảnh)

### `POST /api/compare-thresholds`
So sánh kết quả với các confidence threshold khác nhau

## 🐛 Troubleshooting

### Backend không chạy được

1. Kiểm tra Python version: `python3 --version` (cần 3.8+)
2. Kiểm tra model path trong `backend/app.py`
3. Kiểm tra dependencies: `pip list | grep ultralytics`
4. Xem logs trong terminal để biết lỗi cụ thể

### Frontend không kết nối được backend

1. Đảm bảo backend đang chạy tại `http://localhost:8000`
2. Kiểm tra CORS settings trong `backend/app.py`
3. Kiểm tra API URL trong `frontend/src/services/api.js`

### Model không load được

1. Kiểm tra file `best.pt` có tồn tại không
2. Kiểm tra đường dẫn `MODEL_PATH` trong `backend/app.py`
3. Thử dùng đường dẫn tuyệt đối trong `backend/app.py`

### Node.js version quá cũ

Nếu gặp lỗi với Node.js < 14:

**Cách 1: Sử dụng nvm (Khuyến nghị)**
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install --lts
nvm use --lts
```

**Cách 2: Cài đặt từ NodeSource**
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Port đã được sử dụng

- **Backend**: Đổi port trong `backend/app.py` (dòng cuối)
- **Frontend**: Thêm `PORT=3001` vào `frontend/package.json` scripts

## 📝 Ghi Chú

- File upload được lưu tạm trong `backend/uploads/` và tự động xóa sau khi xử lý
- Model được load một lần khi khởi động backend
- Frontend sử dụng Tailwind CSS cho styling
- ESLint đã được tắt tạm thời để tương thích với Node.js cũ

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.
# Animal-Detection-System-Data-Mining-Project
