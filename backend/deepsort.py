"""
DeepSORT Implementation cho Multi-Object Tracking
Dựa trên paper: "Simple Online and Realtime Tracking with a Deep Association Metric"
"""

import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import cv2


class KalmanBoxTracker:
    """
    Kalman Filter để track bounding box
    State: [x, y, s, r, x', y', s']
    x, y: center coordinates
    s: scale (area)
    r: aspect ratio
    x', y', s': velocities
    """
    
    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2] format
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Covariance matrix
        self.kf.P[4:, 4:] *= 1000.  # High uncertainty for velocities
        self.kf.P *= 10.
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Convert bbox to [x, y, s, r] format
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        
        # Fix shape: kf.x[:4] is (4,1), need to assign as array
        self.kf.x[:4] = np.array([x, y, s, r]).reshape(4, 1)
        
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox):
        """
        Update Kalman filter với bbox mới
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
        
        # Convert bbox to [x, y, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        
        # Fix shape: update expects (4,) or (4,1)
        z = np.array([x, y, s, r], dtype=np.float32)
        self.kf.update(z)
    
    def predict(self):
        """
        Predict next state
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        
        return self.get_state()
    
    def get_state(self):
        """
        Get current bounding box estimate
        Returns: [x1, y1, x2, y2]
        """
        # Fix shape: kf.x[:4] is (4,1), need to flatten
        ret = self.kf.x[:4].copy().flatten()
        x, y, s, r = ret
        
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
        
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class FeatureExtractor:
    """
    Feature Extractor để extract appearance features từ bounding boxes
    Sử dụng lightweight CNN hoặc simple feature extraction
    """
    
    def __init__(self, feature_dim=128):
        """
        feature_dim: dimension của feature vector
        """
        self.feature_dim = feature_dim
        # Sử dụng simple histogram-based features cho bước đầu
        # Có thể nâng cấp lên CNN sau
    
    def extract(self, detections, frame):
        """
        Extract features từ detections
        
        Parameters:
        - detections: list of detections, mỗi detection có 'bbox'
        - frame: numpy array (H, W, 3) RGB image
        
        Returns:
        - features: numpy array (N, feature_dim)
        """
        if len(detections) == 0:
            return np.empty((0, self.feature_dim))
        
        features = []
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clip bbox to frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            # Crop region
            crop = frame_bgr[y1:y2, x1:x2]
            
            if crop.size == 0:
                # Empty crop, use zero features
                features.append(np.zeros(self.feature_dim))
                continue
            
            # Resize to fixed size
            crop_resized = cv2.resize(crop, (64, 128))
            
            # Extract simple features (histogram + HOG-like)
            # Color histogram
            hist_b = cv2.calcHist([crop_resized], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([crop_resized], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([crop_resized], [2], None, [32], [0, 256])
            
            # Normalize
            hist_b = hist_b.flatten() / (hist_b.sum() + 1e-6)
            hist_g = hist_g.flatten() / (hist_g.sum() + 1e-6)
            hist_r = hist_r.flatten() / (hist_r.sum() + 1e-6)
            
            # Combine features
            feature = np.concatenate([hist_b, hist_g, hist_r])
            
            # Pad or truncate to feature_dim
            if len(feature) < self.feature_dim:
                feature = np.pad(feature, (0, self.feature_dim - len(feature)))
            else:
                feature = feature[:self.feature_dim]
            
            # Normalize to unit vector
            feature = feature / (np.linalg.norm(feature) + 1e-6)
            
            features.append(feature)
        
        return np.array(features, dtype=np.float32)


class Track:
    """
    Track object để quản lý một đối tượng được track
    """
    
    def __init__(self, detection, feature, track_id):
        """
        Parameters:
        - detection: dict với keys: bbox, class, confidence
        - feature: feature vector
        - track_id: unique track ID
        """
        self.track_id = track_id
        self.kalman = KalmanBoxTracker(detection['bbox'])
        self.feature = feature / (np.linalg.norm(feature) + 1e-6)  # Normalize
        self.class_name = detection['class']
        self.confidence = detection['confidence']
        self.class_id = detection.get('class_id', 0)
        
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        self.is_confirmed = False
        self.history = deque(maxlen=30)  # Store last 30 states
    
    def update(self, detection, feature):
        """
        Update track với detection mới
        """
        self.kalman.update(detection['bbox'])
        self.feature = feature / (np.linalg.norm(feature) + 1e-6)
        self.class_name = detection['class']
        self.confidence = detection['confidence']
        self.class_id = detection.get('class_id', 0)
        
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
        
        # Confirm track sau min_hits lần update
        if self.hit_streak >= 3:
            self.is_confirmed = True
        
        # Update history
        state = self.kalman.get_state()
        self.history.append(state.copy())
    
    def predict(self):
        """
        Predict next state
        """
        state = self.kalman.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return state
    
    def get_state(self):
        """
        Get current state
        """
        return self.kalman.get_state()
    
    def to_dict(self, is_new=False):
        """
        Convert track to dict format
        """
        bbox = self.get_state()
        return {
            'track_id': self.track_id,
            'bbox': bbox.tolist(),
            'class': self.class_name,
            'class_id': self.class_id,
            'confidence': float(self.confidence),
            'is_new': is_new,
            'age': self.age,
            'hit_streak': self.hit_streak
        }


class DeepSortTracker:
    """
    DeepSORT Tracker chính
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, feature_dim=128):
        """
        Parameters:
        - max_age: số frame tối đa để giữ track không match
        - min_hits: số frame match tối thiểu để confirm track
        - iou_threshold: threshold cho IoU matching
        - feature_dim: dimension của feature vector
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.next_id = 1
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)
        
        # Feature cache để tính cosine similarity
        self.feature_cache = {}
    
    def update(self, detections, frame):
        """
        Update tracker với detections mới
        
        Parameters:
        - detections: list of dicts, mỗi dict có keys: bbox, class, confidence, class_id
        - frame: numpy array (H, W, 3) RGB image
        
        Returns:
        - tracks: list of Track objects
        """
        # Extract features từ detections
        features = self.feature_extractor.extract(detections, frame)
        
        # Predict tracks
        for track in self.tracks:
            track.predict()
        
        # Association
        if len(detections) == 0:
            # Không có detections, chỉ predict
            return self.tracks
        
        if len(self.tracks) == 0:
            # Không có tracks, tạo mới tất cả detections
            for i, det in enumerate(detections):
                track = Track(det, features[i], self.next_id)
                self.tracks.append(track)
                self.next_id += 1
            return self.tracks
        
        # Tính cost matrix
        cost_matrix = self._compute_cost_matrix(detections, features)
        
        # Hungarian algorithm để match
        matched, unmatched_dets, unmatched_trks = self._associate(cost_matrix)
        
        # Update matched tracks
        for m in matched:
            det_idx, trk_idx = m
            self.tracks[trk_idx].update(detections[det_idx], features[det_idx])
        
        # Create new tracks cho unmatched detections
        for i in unmatched_dets:
            track = Track(detections[i], features[i], self.next_id)
            self.tracks.append(track)
            self.next_id += 1
        
        # Delete old tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update < self.max_age and (t.is_confirmed or t.time_since_update < 1)
        ]
        
        return self.tracks
    
    def _compute_cost_matrix(self, detections, features):
        """
        Compute cost matrix cho association
        Cost = IoU distance + Feature distance
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.empty((0, 0))
        
        # IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                iou = self._compute_iou(det['bbox'], track.get_state())
                iou_matrix[i, j] = iou
        
        # Feature distance matrix (cosine distance)
        feature_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
        for i, feat in enumerate(features):
            for j, track in enumerate(self.tracks):
                # Cosine distance = 1 - cosine similarity
                cosine_sim = np.dot(feat, track.feature)
                feature_matrix[i, j] = 1 - cosine_sim
        
        # Combine costs (weighted)
        iou_cost = 1 - iou_matrix  # Convert IoU to cost (higher IoU = lower cost)
        feature_cost = feature_matrix
        
        # Weighted combination
        cost_matrix = 0.5 * iou_cost + 0.5 * feature_cost
        
        return cost_matrix
    
    def _compute_iou(self, bbox1, bbox2):
        """
        Compute IoU giữa 2 bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _associate(self, cost_matrix, threshold=None):
        """
        Associate detections với tracks sử dụng Hungarian algorithm
        
        Returns:
        - matched: list of (det_idx, trk_idx) tuples
        - unmatched_dets: list of unmatched detection indices
        - unmatched_trks: list of unmatched track indices
        """
        if threshold is None:
            threshold = self.iou_threshold
        
        if cost_matrix.size == 0:
            return [], list(range(len(cost_matrix))), list(range(len(cost_matrix[0]))) if cost_matrix.ndim == 2 else []
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_dets = []
        unmatched_trks = []
        
        # Filter matches by threshold
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < threshold:
                matched.append((i, j))
            else:
                unmatched_dets.append(i)
                unmatched_trks.append(j)
        
        # Find unmatched detections và tracks
        all_det_indices = set(range(len(cost_matrix)))
        all_trk_indices = set(range(len(cost_matrix[0])))
        
        matched_det_indices = set([m[0] for m in matched])
        matched_trk_indices = set([m[1] for m in matched])
        
        unmatched_dets.extend(all_det_indices - matched_det_indices)
        unmatched_trks.extend(all_trk_indices - matched_trk_indices)
        
        return matched, unmatched_dets, unmatched_trks
    
    def reset(self):
        """
        Reset tracker (xóa tất cả tracks)
        """
        self.tracks = []
        self.next_id = 1
        self.feature_cache = {}

