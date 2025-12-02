"""
ObjectDetector Class - Wrapper cho YOLO model
Copy từ inference_detection.py với một số điều chỉnh cho API
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
import os


class ObjectDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        model_path: đường dẫn tới model đã train
        conf_threshold: ngưỡng confidence (0-1)
        iou_threshold: ngưỡng IoU cho NMS (Non-Maximum Suppression)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Kiểm tra model tồn tại
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model không tồn tại: {model_path}")

        # Load model
        self.model = YOLO(model_path)
        
        # Lấy thông tin classes
        self.classes = self.model.names

    def detect_image(self, image_path, save_path=None, show=False):
        """
        Nhận diện đối tượng trong một ảnh

        Parameters:
        - image_path: đường dẫn tới ảnh cần nhận diện
        - save_path: đường dẫn lưu ảnh kết quả (None = không lưu)
        - show: hiển thị kết quả bằng matplotlib (True/False)

        Returns:
        - result: kết quả detection từ YOLO
        - img_rgb: ảnh với bounding boxes (RGB format)
        """
        # Kiểm tra file tồn tại
        if not Path(image_path).exists():
            return None, None

        # Chạy inference với error handling
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=False,
                verbose=False
            )

            # Kiểm tra kết quả
            if not results or len(results) == 0:
                return None, None

            # Lấy kết quả đầu tiên
            result = results[0]

            # Kiểm tra result hợp lệ
            if result is None:
                return None, None

            # Vẽ bounding boxes
            try:
                img_with_boxes = result.plot()
            except Exception as e:
                print(f"Error plotting result: {e}")
                # Nếu không vẽ được, đọc ảnh gốc
                img_bgr = cv2.imread(str(image_path))
                if img_bgr is None:
                    return None, None
                img_with_boxes = img_bgr

            # Chuyển từ BGR sang RGB cho web
            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error in detect_image: {e}")
            import traceback
            traceback.print_exc()
            return None, None

        # Lưu ảnh nếu cần
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img_with_boxes)

        return result, img_rgb

    def detect_folder(self, folder_path, output_folder=None):
        """
        Nhận diện tất cả ảnh trong một folder

        Parameters:
        - folder_path: folder chứa ảnh cần nhận diện
        - output_folder: folder lưu kết quả (None = không lưu)

        Returns:
        - results_summary: danh sách kết quả cho mỗi ảnh
        """
        folder = Path(folder_path)
        if not folder.exists():
            return None

        # Tạo output folder nếu cần
        if output_folder:
            output = Path(output_folder)
            output.mkdir(parents=True, exist_ok=True)

        # Lấy danh sách ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
        images = [f for f in folder.iterdir()
                 if f.is_file() and f.suffix.lower() in image_extensions]

        if not images:
            return None

        results_summary = []

        for img_path in images:
            # Nhận diện
            save_path = output / img_path.name if output_folder else None
            result, _ = self.detect_image(img_path, save_path=save_path, show=False)

            if result is not None:
                # Lưu thống kê
                num_detections = len(result.boxes)
                detected_classes = [self.classes[int(box.cls[0])] for box in result.boxes]
                confidences = [float(box.conf[0]) for box in result.boxes]

                results_summary.append({
                    'image': img_path.name,
                    'num_detections': num_detections,
                    'classes': detected_classes,
                    'confidences': confidences
                })

        return results_summary

    def detect_with_custom_threshold(self, image_path, conf_threshold):
        """
        Nhận diện với confidence threshold tùy chỉnh
        """
        old_threshold = self.conf_threshold
        self.conf_threshold = conf_threshold
        result, img = self.detect_image(image_path, show=False)
        self.conf_threshold = old_threshold
        return result, img

    def compare_thresholds(self, image_path, thresholds=[0.1, 0.25, 0.5, 0.75]):
        """
        So sánh kết quả với các confidence threshold khác nhau
        """
        results = {}

        for threshold in thresholds:
            result, _ = self.detect_with_custom_threshold(image_path, threshold)

            if result is not None:
                num_detections = len(result.boxes)
                results[threshold] = {
                    'count': num_detections,
                    'classes': [self.classes[int(box.cls[0])] for box in result.boxes]
                }

        return results

