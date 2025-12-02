#!/usr/bin/env python3
"""
Script test toàn bộ hệ thống Object Detection với DeepSORT
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_imports():
    """Test tất cả imports"""
    print("=" * 60)
    print("🧪 TEST 1: Imports")
    print("=" * 60)
    
    tests = [
        ("numpy", "import numpy as np"),
        ("pydantic", "import pydantic"),
        ("scipy", "import scipy"),
        ("filterpy", "from filterpy.kalman import KalmanFilter"),
        ("cv2", "import cv2"),
        ("ultralytics", "from ultralytics import YOLO"),
        ("ObjectDetector", "from inference import ObjectDetector"),
        ("DeepSORT", "from deepsort import DeepSortTracker, KalmanBoxTracker"),
        ("VideoTracker", "from tracker import VideoTracker"),
    ]
    
    results = []
    for name, code in tests:
        try:
            exec(code)
            print(f"✅ {name}: OK")
            results.append(True)
        except Exception as e:
            print(f"❌ {name}: {e}")
            results.append(False)
    
    print()
    return all(results)


def test_kalman_filter():
    """Test Kalman Filter"""
    print("=" * 60)
    print("🧪 TEST 2: Kalman Filter")
    print("=" * 60)
    
    try:
        from deepsort import KalmanBoxTracker
        import numpy as np
        
        # Test init
        tracker = KalmanBoxTracker([100, 100, 200, 200])
        print("✅ KalmanBoxTracker init: OK")
        
        # Test get_state
        state = tracker.get_state()
        assert state.shape == (4,), f"Expected shape (4,), got {state.shape}"
        print(f"✅ get_state: OK (shape: {state.shape})")
        
        # Test update
        tracker.update([110, 110, 210, 210])
        print("✅ update: OK")
        
        # Test predict
        predicted = tracker.predict()
        assert predicted.shape == (4,), f"Expected shape (4,), got {predicted.shape}"
        print("✅ predict: OK")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Kalman Filter test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_deepsort_tracker():
    """Test DeepSORT Tracker"""
    print("=" * 60)
    print("🧪 TEST 3: DeepSORT Tracker")
    print("=" * 60)
    
    try:
        from deepsort import DeepSortTracker
        import numpy as np
        import cv2
        
        # Create mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create tracker
        tracker = DeepSortTracker(max_age=30, min_hits=3)
        print("✅ DeepSortTracker init: OK")
        
        # Test với detections
        detections = [
            {'bbox': [100, 100, 200, 200], 'class': 'person', 'class_id': 0, 'confidence': 0.85},
            {'bbox': [300, 300, 400, 400], 'class': 'car', 'class_id': 2, 'confidence': 0.90}
        ]
        
        tracks = tracker.update(detections, frame)
        print(f"✅ update with detections: OK ({len(tracks)} tracks created)")
        
        # Test với empty detections
        tracks = tracker.update([], frame)
        print(f"✅ update with empty detections: OK ({len(tracks)} tracks)")
        
        # Test predict tracks
        if len(tracker.tracks) > 0:
            for track in tracker.tracks:
                track.predict()
            print("✅ predict tracks: OK")
        
        print()
        return True
    except Exception as e:
        print(f"❌ DeepSORT Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_video_tracker():
    """Test VideoTracker (cần model file)"""
    print("=" * 60)
    print("🧪 TEST 4: VideoTracker")
    print("=" * 60)
    
    # Check model file
    model_paths = ["../best.pt", "./best.pt", "best.pt"]
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("⚠️  Model file not found, skipping VideoTracker test")
        print("   (This is OK if you just want to test imports)")
        print()
        return True
    
    try:
        from tracker import VideoTracker
        
        tracker = VideoTracker(model_path, conf_threshold=0.25, iou_threshold=0.45)
        print(f"✅ VideoTracker init: OK (model: {model_path})")
        print(f"   Classes: {len(tracker.detector.classes)} classes")
        
        print()
        return True
    except Exception as e:
        print(f"❌ VideoTracker test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_tracking_pipeline():
    """Test full tracking pipeline"""
    print("=" * 60)
    print("🧪 TEST 5: Full Tracking Pipeline")
    print("=" * 60)
    
    try:
        from deepsort import DeepSortTracker
        import numpy as np
        import cv2
        
        # Create tracker
        tracker = DeepSortTracker(max_age=30, min_hits=3)
        
        # Create mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Frame 1: Create track
        detections1 = [
            {'bbox': [100, 100, 200, 200], 'class': 'person', 'class_id': 0, 'confidence': 0.85}
        ]
        tracks1 = tracker.update(detections1, frame)
        assert len(tracks1) == 1, f"Expected 1 track, got {len(tracks1)}"
        assert tracks1[0].track_id == 1, f"Expected track_id 1, got {tracks1[0].track_id}"
        print("✅ Frame 1: Track created (ID=1)")
        
        # Frame 2: Update track
        detections2 = [
            {'bbox': [110, 110, 210, 210], 'class': 'person', 'class_id': 0, 'confidence': 0.87}
        ]
        tracks2 = tracker.update(detections2, frame)
        assert len(tracks2) == 1, f"Expected 1 track, got {len(tracks2)}"
        assert tracks2[0].track_id == 1, f"Track ID should remain 1"
        assert tracks2[0].hit_streak == 1, f"Expected hit_streak 1, got {tracks2[0].hit_streak}"
        print("✅ Frame 2: Track updated (ID=1, hit_streak=1)")
        
        # Frame 3: New object
        detections3 = [
            {'bbox': [110, 110, 210, 210], 'class': 'person', 'class_id': 0, 'confidence': 0.87},
            {'bbox': [300, 300, 400, 400], 'class': 'car', 'class_id': 2, 'confidence': 0.90}
        ]
        tracks3 = tracker.update(detections3, frame)
        assert len(tracks3) == 2, f"Expected 2 tracks, got {len(tracks3)}"
        track_ids = [t.track_id for t in tracks3]
        assert 1 in track_ids and 2 in track_ids, f"Expected track IDs 1 and 2, got {track_ids}"
        print("✅ Frame 3: New track created (ID=2)")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Tracking pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🚀 SYSTEM TEST - Object Detection với DeepSORT")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Kalman Filter", test_kalman_filter()))
    results.append(("DeepSORT Tracker", test_deepsort_tracker()))
    results.append(("VideoTracker", test_video_tracker()))
    results.append(("Tracking Pipeline", test_tracking_pipeline()))
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    if failed == 0:
        print("🎉 All tests passed! System is ready to run.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

