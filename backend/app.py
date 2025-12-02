"""
FastAPI Backend cho Object Detection System
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import base64
import tempfile
import uuid
import socket
import asyncio
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
from typing import List, Optional

from inference import ObjectDetector
from tracker import VideoTracker
import time
from collections import defaultdict

# Khởi tạo FastAPI app
app = FastAPI(title="Object Detection API", version="1.0.0")

# CORS middleware - Cải thiện security
# Lấy allowed origins từ environment variable hoặc default
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    # Production: giới hạn origins từ env variable
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    # Development: cho phép localhost và common development origins
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
    ]
    # Nếu có IP trong hostname, thêm vào
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if local_ip and local_ip != "127.0.0.1":
            allowed_origins.extend([
                f"http://{local_ip}:3000",
                f"http://{local_ip}:5173",
            ])
    except:
        pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Khởi tạo detector (model path relative từ thư mục backend)
# Thử nhiều đường dẫn để tìm model
MODEL_PATH = None
possible_paths = [
    "../best.pt",  # Relative từ backend/ lên root
    "./best.pt",   # Nếu chạy từ root
    "best.pt",     # Trong cùng thư mục
]

for path in possible_paths:
    if Path(path).exists():
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Không tìm thấy model best.pt. Vui lòng đảm bảo file best.pt "
        "nằm trong thư mục gốc của dự án hoặc chỉnh MODEL_PATH trong app.py"
    )

try:
    detector = ObjectDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    detector = None

# Session management cho video tracking
# Lưu trữ tracker instances theo session_id
tracker_sessions = {}
SESSION_TIMEOUT = 300  # 5 phút timeout
session_last_activity = {}

def cleanup_old_sessions():
    """Cleanup sessions không hoạt động quá lâu"""
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, last_time in session_last_activity.items():
        if current_time - last_time > SESSION_TIMEOUT:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        if session_id in tracker_sessions:
            del tracker_sessions[session_id]
        if session_id in session_last_activity:
            del session_last_activity[session_id]
    
    return len(sessions_to_remove)


def numpy_to_base64(img_rgb: np.ndarray) -> str:
    """Convert numpy array (RGB) to base64 string"""
    img_pil = Image.fromarray(img_rgb)
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_base64}"


def extract_detections(result, classes):
    """Extract detection data from YOLO result"""
    detections = []
    
    if result is None or not hasattr(result, 'boxes'):
        return detections
    
    boxes = result.boxes
    
    # Kiểm tra nếu không có boxes
    if boxes is None or len(boxes) == 0:
        return detections
    
    try:
        for i, box in enumerate(boxes, 1):
            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Kiểm tra class_id hợp lệ
                if cls_id not in classes:
                    continue
                
                class_name = classes[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Validate bbox coordinates
                if x2 <= x1 or y2 <= y1:
                    continue
                
                detections.append({
                    "id": i,
                    "class": class_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                })
            except (IndexError, ValueError, AttributeError) as e:
                # Skip invalid boxes
                print(f"Warning: Skipping invalid box {i}: {e}")
                continue
    
    except Exception as e:
        print(f"Error extracting detections: {e}")
        return detections
    
    return detections


@app.get("/")
async def root():
    return {
        "message": "Object Detection API",
        "status": "running",
        "model_loaded": detector is not None
    }


@app.get("/api/model-info")
async def get_model_info():
    """Lấy thông tin model"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải. Vui lòng kiểm tra lại server.")
    
    classes_dict = {str(k): v for k, v in detector.classes.items()}
    
    return {
        "model_path": MODEL_PATH,
        "num_classes": len(detector.classes),
        "classes": classes_dict,
        "default_conf_threshold": detector.conf_threshold,
        "default_iou_threshold": detector.iou_threshold
    }


@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    """
    Nhận diện đối tượng trong 1 ảnh
    
    Parameters:
    - file: File ảnh upload
    - conf_threshold: Confidence threshold (0-1)
    - iou_threshold: IoU threshold (0-1)
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải. Vui lòng kiểm tra lại server.")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File phải là ảnh. Vui lòng chọn file ảnh hợp lệ.")
    
    # Validate thresholds
    if not (0 <= conf_threshold <= 1):
        raise HTTPException(status_code=400, detail="Ngưỡng confidence phải trong khoảng 0 đến 1.")
    if not (0 <= iou_threshold <= 1):
        raise HTTPException(status_code=400, detail="Ngưỡng IoU phải trong khoảng 0 đến 1.")
    
    # Validate file size (max 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    try:
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File trống. Vui lòng chọn file ảnh hợp lệ.")
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"Kích thước file vượt quá giới hạn {MAX_FILE_SIZE / 1024 / 1024}MB. Vui lòng chọn ảnh nhỏ hơn.")
        
        # Reset file pointer
        await file.seek(0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {str(e)}")
    
    # Lưu file tạm - Sử dụng tempfile để an toàn hơn
    # Sanitize filename để tránh path traversal
    safe_filename = Path(file.filename).name if file.filename else "image.jpg"
    # Loại bỏ các ký tự không hợp lệ
    safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
    if not safe_filename or not safe_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')):
        safe_filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
    
    # Sử dụng tempfile để đảm bảo an toàn
    temp_dir = Path(tempfile.gettempdir()) / "object_detection_uploads"
    temp_dir.mkdir(exist_ok=True, mode=0o700)  # Chỉ owner có quyền
    
    temp_path = temp_dir / safe_filename
    
    try:
        # Lưu file (đã đọc file_content ở trên)
        with open(temp_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Detect với thresholds (không thay đổi global state)
        # Tạo temporary detector config hoặc pass thresholds vào detect_image
        # Tạm thời: lưu old thresholds và restore sau
        old_conf = detector.conf_threshold
        old_iou = detector.iou_threshold
        
        try:
            detector.conf_threshold = conf_threshold
            detector.iou_threshold = iou_threshold
            
            # Detect với timeout
            try:
                result, img_rgb = await asyncio.wait_for(
                    asyncio.to_thread(detector.detect_image, str(temp_path), None, False),
                    timeout=30.0  # 30 seconds timeout
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Hết thời gian xử lý. Ảnh xử lý quá lâu. Vui lòng thử với ảnh nhỏ hơn.")
        finally:
            # Restore old thresholds để tránh race condition
            detector.conf_threshold = old_conf
            detector.iou_threshold = old_iou
        
        if result is None:
            raise HTTPException(status_code=400, detail="Không thể xử lý ảnh. Vui lòng kiểm tra file ảnh có hợp lệ không.")
        
        # Extract detections
        detections = extract_detections(result, detector.classes)
        
        # Convert image to base64
        image_base64 = numpy_to_base64(img_rgb)
        
        # Calculate statistics
        if detections:
            confidences = [d["confidence"] for d in detections]
            statistics = {
                "total": len(detections),
                "classes": list(set([d["class"] for d in detections])),
                "avg_confidence": round(np.mean(confidences), 4),
                "min_confidence": round(np.min(confidences), 4),
                "max_confidence": round(np.max(confidences), 4)
            }
        else:
            statistics = {
                "total": 0,
                "classes": [],
                "avg_confidence": 0,
                "min_confidence": 0,
                "max_confidence": 0
            }
        
        return {
            "success": True,
            "detections": detections,
            "image_base64": image_base64,
            "statistics": statistics
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions (đã có detail message)
        raise
    except Exception as e:
        # Log error để debug
        import traceback
        print(f"Error processing image: {e}")
        print(traceback.format_exc())
        # Trả về error message user-friendly
        error_msg = str(e)
        if "CUDA" in error_msg or "GPU" in error_msg:
            error_msg = "Lỗi GPU. Vui lòng thử lại hoặc kiểm tra cấu hình GPU."
        elif "memory" in error_msg.lower() or "Memory" in error_msg:
            error_msg = "Không đủ bộ nhớ để xử lý ảnh. Vui lòng thử với ảnh nhỏ hơn."
        elif "format" in error_msg.lower() or "decode" in error_msg.lower():
            error_msg = "Không thể đọc file ảnh. Vui lòng kiểm tra định dạng file."
        else:
            error_msg = f"Lỗi xử lý ảnh: {error_msg}"
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Xóa file tạm
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    """
    Nhận diện nhiều ảnh cùng lúc
    NOTE: Endpoint này vẫn được giữ lại để tương thích, nhưng frontend hiện tại không sử dụng
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải. Vui lòng kiểm tra lại server.")
    
    # Validate thresholds
    if not (0 <= conf_threshold <= 1):
        raise HTTPException(status_code=400, detail="Ngưỡng confidence phải trong khoảng 0 đến 1.")
    if not (0 <= iou_threshold <= 1):
        raise HTTPException(status_code=400, detail="Ngưỡng IoU phải trong khoảng 0 đến 1.")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Tối đa 20 ảnh mỗi batch.")
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Lưu old thresholds và restore sau để tránh race condition
    old_conf = detector.conf_threshold
    old_iou = detector.iou_threshold
    
    try:
        detector.conf_threshold = conf_threshold
        detector.iou_threshold = iou_threshold
        
        # Sử dụng tempfile cho batch processing
        temp_dir = Path(tempfile.gettempdir()) / "object_detection_uploads"
        temp_dir.mkdir(exist_ok=True, mode=0o700)
        
        results = []
        all_classes = []
        all_confidences = []
        
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                continue
            
            # Validate file size
            file_content = await file.read()
            if len(file_content) > MAX_FILE_SIZE:
                print(f"Skipping {file.filename}: file too large")
                continue
            
            # Reset file pointer
            await file.seek(0)
            
            # Sanitize filename
            safe_filename = Path(file.filename).name if file.filename else f"image_{uuid.uuid4().hex[:8]}.jpg"
            safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
            if not safe_filename or not safe_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')):
                safe_filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
            
            temp_path = temp_dir / safe_filename
            
            try:
                # Lưu file
                with open(temp_path, "wb") as buffer:
                    buffer.write(file_content)
                
                # Detect với timeout
                try:
                    result, img_rgb = await asyncio.wait_for(
                        asyncio.to_thread(detector.detect_image, str(temp_path), None, False),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    print(f"Timeout processing {safe_filename}")
                    continue
                
                if result is not None:
                    detections = extract_detections(result, detector.classes)
                    image_base64 = numpy_to_base64(img_rgb)
                    
                    # Collect statistics
                    for det in detections:
                        all_classes.append(det["class"])
                        all_confidences.append(det["confidence"])
                    
                    results.append({
                        "filename": safe_filename,
                        "detections": detections,
                        "num_detections": len(detections),
                        "image_base64": image_base64
                    })
                
                # Xóa file tạm
                if temp_path.exists():
                    temp_path.unlink()
            
            except Exception as e:
                print(f"Error processing {safe_filename}: {e}")
                # Xóa file tạm nếu có lỗi
                if temp_path.exists():
                    temp_path.unlink()
                continue
        
            # Summary statistics
        from collections import Counter
        class_distribution = dict(Counter(all_classes))
        
        summary = {
            "total_images": len(results),
            "images_with_detections": sum(1 for r in results if r["num_detections"] > 0),
            "total_detections": sum(r["num_detections"] for r in results),
            "class_distribution": class_distribution,
            "avg_confidence": round(np.mean(all_confidences), 4) if all_confidences else 0
        }
        
        return {
            "success": True,
            "results": results,
            "summary": summary
        }
    
    finally:
        # Restore thresholds
        detector.conf_threshold = old_conf
        detector.iou_threshold = old_iou


@app.post("/api/compare-thresholds")
async def compare_thresholds(
    file: UploadFile = File(...),
    thresholds: str = Form("[0.1, 0.25, 0.5, 0.75]")
):
    """
    So sánh kết quả với các confidence threshold khác nhau
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    import json
    try:
        threshold_list = json.loads(thresholds)
        # Validate input
        if not isinstance(threshold_list, list):
            raise ValueError("Thresholds must be a list")
        if len(threshold_list) > 10:  # Giới hạn số lượng thresholds
            raise HTTPException(status_code=400, detail="Tối đa 10 thresholds được phép so sánh.")
        # Validate mỗi threshold
        for t in threshold_list:
            if not isinstance(t, (int, float)) or not (0 <= t <= 1):
                raise HTTPException(status_code=400, detail=f"Threshold {t} không hợp lệ. Phải trong khoảng 0 đến 1.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Định dạng JSON không hợp lệ cho thresholds.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        threshold_list = [0.1, 0.25, 0.5, 0.75]  # Fallback
    
    # Sử dụng tempfile và sanitize filename
    safe_filename = Path(file.filename).name if file.filename else f"image_{uuid.uuid4().hex[:8]}.jpg"
    safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
    if not safe_filename or not safe_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')):
        safe_filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
    
    temp_dir = Path(tempfile.gettempdir()) / "object_detection_uploads"
    temp_dir.mkdir(exist_ok=True, mode=0o700)
    temp_path = temp_dir / safe_filename
    
    try:
        # Lưu file
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Compare thresholds với timeout
        try:
            comparisons = await asyncio.wait_for(
                asyncio.to_thread(detector.compare_thresholds, str(temp_path), threshold_list),
                timeout=60.0  # 60s cho multiple thresholds
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Hết thời gian xử lý. So sánh thresholds quá lâu. Vui lòng thử với ít thresholds hơn.")
        
        # Format output
        result = {}
        for threshold, data in comparisons.items():
            result[str(threshold)] = {
                "count": data["count"],
                "classes": list(set(data["classes"]))
            }
        
        return {
            "success": True,
            "comparisons": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            error_msg = "Hết thời gian xử lý. Vui lòng thử lại với ảnh nhỏ hơn."
        elif "memory" in error_msg.lower():
            error_msg = "Không đủ bộ nhớ để xử lý. Vui lòng thử với ảnh nhỏ hơn."
        else:
            error_msg = f"Lỗi xử lý: {error_msg}"
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    session_id: Optional[str] = Form(None)
):
    """
    Nhận diện và tracking đối tượng trong video frame
    
    Parameters:
    - file: File ảnh frame upload
    - conf_threshold: Confidence threshold (0-1)
    - iou_threshold: IoU threshold (0-1)
    - session_id: Session ID để maintain tracking state (optional)
    
    Returns:
    - tracks: List of tracks với ID cố định
    - image_base64: Image với bounding boxes và track IDs
    - statistics: Thống kê về tracks
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải. Vui lòng kiểm tra lại server.")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File phải là ảnh. Vui lòng chọn file ảnh hợp lệ.")
    
    # Validate thresholds
    if not (0 <= conf_threshold <= 1):
        raise HTTPException(status_code=400, detail="Ngưỡng confidence phải trong khoảng 0 đến 1.")
    if not (0 <= iou_threshold <= 1):
        raise HTTPException(status_code=400, detail="Ngưỡng IoU phải trong khoảng 0 đến 1.")
    
    # Validate file size (max 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    try:
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File trống. Vui lòng chọn file ảnh hợp lệ.")
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"Kích thước file vượt quá giới hạn {MAX_FILE_SIZE / 1024 / 1024}MB. Vui lòng chọn ảnh nhỏ hơn.")
        
        await file.seek(0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file: {str(e)}")
    
    # Cleanup old sessions
    cleanup_old_sessions()
    
    # Get or create session
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    # Update session activity
    session_last_activity[session_id] = time.time()
    
    # Get or create tracker for this session
    if session_id not in tracker_sessions:
        try:
            tracker_sessions[session_id] = VideoTracker(
                model_path=MODEL_PATH,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            print(f"✅ Created new tracker session: {session_id}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi khởi tạo tracker: {str(e)}")
    
    tracker = tracker_sessions[session_id]
    
    # Update tracker thresholds nếu thay đổi
    if tracker.detector.conf_threshold != conf_threshold:
        tracker.detector.conf_threshold = conf_threshold
    if tracker.detector.iou_threshold != iou_threshold:
        tracker.detector.iou_threshold = iou_threshold
    
    # Sanitize filename
    safe_filename = Path(file.filename).name if file.filename else "image.jpg"
    safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._-")
    if not safe_filename or not safe_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')):
        safe_filename = f"frame_{uuid.uuid4().hex[:8]}.jpg"
    
    # Sử dụng tempfile
    temp_dir = Path(tempfile.gettempdir()) / "object_detection_uploads"
    temp_dir.mkdir(exist_ok=True, mode=0o700)
    temp_path = temp_dir / safe_filename
    
    try:
        # Lưu file
        with open(temp_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Process frame với timeout
        try:
            result, img_rgb, tracks = await asyncio.wait_for(
                asyncio.to_thread(tracker.process_frame, str(temp_path)),
                timeout=30.0  # 30 seconds timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Hết thời gian xử lý. Frame xử lý quá lâu. Vui lòng thử với frame nhỏ hơn.")
        
        if result is None:
            raise HTTPException(status_code=400, detail="Không thể xử lý frame. Vui lòng kiểm tra file ảnh có hợp lệ không.")
        
        # Convert image to base64
        image_base64 = numpy_to_base64(img_rgb)
        
        # Calculate statistics
        if tracks:
            new_tracks = [t for t in tracks if t.get("is_new", False)]
            active_tracks = [t for t in tracks if not t.get("is_new", False)]
            
            confidences = [t["confidence"] for t in tracks]
            classes = list(set([t["class"] for t in tracks]))
            
            statistics = {
                "total_tracks": len(tracks),
                "new_tracks": len(new_tracks),
                "active_tracks": len(active_tracks),
                "classes": classes,
                "avg_confidence": round(np.mean(confidences), 4) if confidences else 0,
                "min_confidence": round(np.min(confidences), 4) if confidences else 0,
                "max_confidence": round(np.max(confidences), 4) if confidences else 0
            }
        else:
            statistics = {
                "total_tracks": 0,
                "new_tracks": 0,
                "active_tracks": 0,
                "classes": [],
                "avg_confidence": 0,
                "min_confidence": 0,
                "max_confidence": 0
            }
        
        return {
            "success": True,
            "tracks": tracks,
            "image_base64": image_base64,
            "statistics": statistics,
            "session_id": session_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error processing video frame: {e}")
        print(traceback.format_exc())
        error_msg = str(e)
        if "CUDA" in error_msg or "GPU" in error_msg:
            error_msg = "Lỗi GPU. Vui lòng thử lại hoặc kiểm tra cấu hình GPU."
        elif "memory" in error_msg.lower() or "Memory" in error_msg:
            error_msg = "Không đủ bộ nhớ để xử lý frame. Vui lòng thử với frame nhỏ hơn."
        elif "format" in error_msg.lower() or "decode" in error_msg.lower():
            error_msg = "Không thể đọc file ảnh. Vui lòng kiểm tra định dạng file."
        else:
            error_msg = f"Lỗi xử lý frame: {error_msg}"
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Xóa file tạm
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/reset-tracking-session")
async def reset_tracking_session(session_id: str = Form(...)):
    """
    Reset tracking session (xóa tất cả tracks)
    """
    if session_id in tracker_sessions:
        tracker_sessions[session_id].reset()
        return {"success": True, "message": f"Session {session_id} đã được reset"}
    else:
        return {"success": False, "message": f"Session {session_id} không tồn tại"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

