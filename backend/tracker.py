"""
VideoTracker - Wrapper để tích hợp YOLO Detection + DeepSORT Tracking
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from inference import ObjectDetector
from deepsort import DeepSortTracker


class VideoTracker:
    """
    Video Tracker kết hợp YOLO detection và DeepSORT tracking
    """
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45, 
                 max_age=30, min_hits=3, track_iou_threshold=0.3):
        """
        Parameters:
        - model_path: đường dẫn tới YOLO model
        - conf_threshold: confidence threshold cho YOLO
        - iou_threshold: IoU threshold cho YOLO NMS
        - max_age: số frame tối đa để giữ track không match
        - min_hits: số frame match tối thiểu để confirm track
        - track_iou_threshold: IoU threshold cho tracking association
        """
        # YOLO Detector
        self.detector = ObjectDetector(model_path, conf_threshold, iou_threshold)
        
        # DeepSORT Tracker
        self.tracker = DeepSortTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=track_iou_threshold
        )
    
    def process_frame(self, frame_image_path: str, frame_image_array: Optional[np.ndarray] = None):
        """
        Process một frame: Detect + Track
        
        Parameters:
        - frame_image_path: đường dẫn tới frame image (nếu có)
        - frame_image_array: numpy array của frame (H, W, 3) RGB (nếu có)
        
        Returns:
        - result: YOLO result object
        - img_rgb: image với bounding boxes và track IDs (RGB format)
        - tracks: list of track dicts
        """
        # Load frame
        if frame_image_array is not None:
            img_rgb = frame_image_array.copy()
            # Convert RGB to BGR cho YOLO (YOLO expects BGR)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # Save temp để YOLO có thể đọc
            temp_path = Path(frame_image_path) if frame_image_path else None
            if temp_path and temp_path.exists():
                cv2.imwrite(str(temp_path), img_bgr)
        else:
            if not Path(frame_image_path).exists():
                return None, None, []
            
            # Read image
            img_bgr = cv2.imread(str(frame_image_path))
            if img_bgr is None:
                return None, None, []
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. YOLO Detection
        try:
            results = self.detector.model.predict(
                source=img_bgr,
                conf=self.detector.conf_threshold,
                iou=self.detector.iou_threshold,
                save=False,
                verbose=False
            )
            
            if not results or len(results) == 0:
                return None, img_rgb, []
            
            result = results[0]
            
            if result is None or not hasattr(result, 'boxes') or result.boxes is None:
                return None, img_rgb, []
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return None, img_rgb, []
        
        # 2. Extract detections format cho DeepSORT
        detections = self._extract_detections_for_tracking(result)
        
        if len(detections) == 0:
            # Không có detections, chỉ predict tracks
            tracks = self.tracker.update([], img_rgb)
            img_with_tracks = self._draw_tracks(img_rgb, tracks, is_new_tracks={})
            return result, img_with_tracks, []
        
        # 3. DeepSORT Tracking
        tracks = self.tracker.update(detections, img_rgb)
        
        # 4. Format tracks output
        formatted_tracks = self._format_tracks(tracks, detections)
        
        # 5. Draw tracks on image
        is_new_tracks = {t['track_id']: t['is_new'] for t in formatted_tracks}
        img_with_tracks = self._draw_tracks(img_rgb, tracks, is_new_tracks)
        
        return result, img_with_tracks, formatted_tracks
    
    def _extract_detections_for_tracking(self, result) -> List[Dict]:
        """
        Extract detections từ YOLO result format cho DeepSORT
        
        Returns:
        - detections: list of dicts với keys: bbox, class, confidence, class_id
        """
        detections = []
        
        if result is None or not hasattr(result, 'boxes'):
            return detections
        
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return detections
        
        try:
            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Kiểm tra class_id hợp lệ
                    if cls_id not in self.detector.classes:
                        continue
                    
                    class_name = self.detector.classes[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Validate bbox coordinates
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'class': class_name,
                        'class_id': cls_id,
                        'confidence': float(conf)
                    })
                except (IndexError, ValueError, AttributeError) as e:
                    print(f"Warning: Skipping invalid box: {e}")
                    continue
        
        except Exception as e:
            print(f"Error extracting detections: {e}")
            return detections
        
        return detections
    
    def _format_tracks(self, tracks, detections) -> List[Dict]:
        """
        Format tracks thành dict format cho API response
        
        Returns:
        - formatted_tracks: list of dicts
        """
        formatted_tracks = []
        
        # Track which detections are new (first time seen)
        # We'll mark tracks as new if they were just created
        track_ids_seen = set()
        
        for track in tracks:
            # Check if this is a new track (just created)
            is_new = track.hit_streak == 1 and track.age == 1
            
            track_dict = track.to_dict(is_new=is_new)
            formatted_tracks.append(track_dict)
            track_ids_seen.add(track.track_id)
        
        return formatted_tracks
    
    def _draw_tracks(self, img_rgb: np.ndarray, tracks, is_new_tracks: Dict[int, bool]) -> np.ndarray:
        """
        Vẽ bounding boxes và track IDs lên image
        
        Parameters:
        - img_rgb: RGB image
        - tracks: list of Track objects
        - is_new_tracks: dict mapping track_id to is_new flag
        
        Returns:
        - img_with_tracks: image với bounding boxes và IDs
        """
        img_with_tracks = img_rgb.copy()
        
        for track in tracks:
            bbox = track.get_state()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clip to image bounds
            h, w = img_with_tracks.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            # Color based on confidence và is_new
            confidence = track.confidence
            is_new = is_new_tracks.get(track.track_id, False)
            
            if is_new:
                # New track: bright green
                color = (0, 255, 0)  # Green
            elif confidence > 0.7:
                # High confidence: blue
                color = (255, 0, 0)  # Blue (BGR format)
            elif confidence > 0.5:
                # Medium confidence: yellow
                color = (0, 255, 255)  # Yellow
            else:
                # Low confidence: red
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(img_with_tracks, (x1, y1), (x2, y2), color, 2)
            
            # Label với track ID và class
            label = f"ID:{track.track_id} {track.class_name} {int(confidence * 100)}%"
            if is_new:
                label = "NEW " + label
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw label background
            label_y = max(y1, text_height + 5)
            cv2.rectangle(
                img_with_tracks,
                (x1, label_y - text_height - 5),
                (x1 + text_width + 5, label_y + baseline),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_with_tracks,
                label,
                (x1 + 2, label_y),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness,
                cv2.LINE_AA
            )
        
        return img_with_tracks
    
    def reset(self):
        """
        Reset tracker (xóa tất cả tracks)
        """
        self.tracker.reset()
    
    def get_active_tracks_count(self):
        """
        Get số lượng tracks đang active
        """
        return len([t for t in self.tracker.tracks if t.time_since_update < self.tracker.max_age])

