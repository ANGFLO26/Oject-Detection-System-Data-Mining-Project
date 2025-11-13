"""
FastAPI Backend cho Animal Detection System
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import base64
import tempfile
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
from typing import List, Optional

from inference import AnimalDetector

# Khởi tạo FastAPI app
app = FastAPI(title="Animal Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "nằm trong thư mục the_end/ hoặc chỉnh MODEL_PATH trong app.py"
    )

try:
    detector = AnimalDetector(
        model_path=MODEL_PATH,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    detector = None


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
    boxes = result.boxes
    
    for i, box in enumerate(boxes, 1):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = classes[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        detections.append({
            "id": i,
            "class": class_name,
            "class_id": cls_id,
            "confidence": round(conf, 4),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "width": float(x2 - x1),
            "height": float(y2 - y1)
        })
    
    return detections


@app.get("/")
async def root():
    return {
        "message": "Animal Detection API",
        "status": "running",
        "model_loaded": detector is not None
    }


@app.get("/api/model-info")
async def get_model_info():
    """Lấy thông tin model"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    classes_dict = {str(k): v for k, v in detector.classes.items()}
    
    return {
        "model_path": MODEL_PATH,
        "num_classes": len(detector.classes),
        "classes": classes_dict,
        "default_conf_threshold": detector.conf_threshold,
        "default_iou_threshold": detector.iou_threshold
    }


@app.post("/api/detect")
async def detect_animal(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45)
):
    """
    Nhận diện động vật trong 1 ảnh
    
    Parameters:
    - file: File ảnh upload
    - conf_threshold: Confidence threshold (0-1)
    - iou_threshold: IoU threshold (0-1)
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Lưu file tạm
    temp_dir = Path("uploads")
    temp_dir.mkdir(exist_ok=True)
    
    temp_path = temp_dir / file.filename
    
    try:
        # Lưu file
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Cập nhật thresholds
        detector.conf_threshold = conf_threshold
        detector.iou_threshold = iou_threshold
        
        # Detect
        result, img_rgb = detector.detect_image(
            image_path=str(temp_path),
            show=False
        )
        
        if result is None:
            raise HTTPException(status_code=400, detail="Could not process image")
        
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
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
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch")
    
    # Cập nhật thresholds
    detector.conf_threshold = conf_threshold
    detector.iou_threshold = iou_threshold
    
    temp_dir = Path("uploads")
    temp_dir.mkdir(exist_ok=True)
    
    results = []
    all_classes = []
    all_confidences = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        temp_path = temp_dir / file.filename
        
        try:
            # Lưu file
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Detect
            result, img_rgb = detector.detect_image(
                image_path=str(temp_path),
                show=False
            )
            
            if result is not None:
                detections = extract_detections(result, detector.classes)
                image_base64 = numpy_to_base64(img_rgb)
                
                # Collect statistics
                for det in detections:
                    all_classes.append(det["class"])
                    all_confidences.append(det["confidence"])
                
                results.append({
                    "filename": file.filename,
                    "detections": detections,
                    "num_detections": len(detections),
                    "image_base64": image_base64
                })
            
            # Xóa file tạm
            if temp_path.exists():
                temp_path.unlink()
        
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
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
    except:
        threshold_list = [0.1, 0.25, 0.5, 0.75]
    
    temp_dir = Path("uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    
    try:
        # Lưu file
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Compare thresholds
        comparisons = detector.compare_thresholds(
            image_path=str(temp_path),
            thresholds=threshold_list
        )
        
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

