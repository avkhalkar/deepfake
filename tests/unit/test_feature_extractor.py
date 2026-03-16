"""
Unit tests for the FeatureExtractor component.

This suite processes dummy video frames and bounding boxes to guarantee that
the generated feature tensors perfectly align with the expected Phase 1 schema
(face_crops, residual_maps, rppg_signals, landmarks, pose_angles and their
dimensional shapes).

Usage:
    pytest tests/unit/test_feature_extractor.py -v
"""

import numpy as np
from src.feature_extractor import FeatureExtractor

def test_feature_extractor_shapes():
    extractor = FeatureExtractor(target_size=(224, 224))
    
    # Create 10 dummy frames of 720p resolution
    num_frames = 10
    frames = np.random.randint(0, 255, (num_frames, 720, 1280, 3), dtype=np.uint8)
    
    # Create 10 dummy bounding boxes
    bboxes = np.array([
        [100, 100, 300, 300],
        [105, 105, 305, 305],
        [110, 110, 310, 310],
        [np.nan, np.nan, np.nan, np.nan], # Simulate dropped frame
        [120, 120, 320, 320],
        [125, 125, 325, 325],
        [130, 130, 330, 330],
        [135, 135, 335, 335],
        [140, 140, 340, 340],
        [145, 145, 345, 345]
    ])
    
    results = extractor.process_sequence(frames, bboxes)
    
    # Verify outputs match Phase 1 Schema
    assert "face_crops" in results
    assert "residual_maps" in results
    assert "rppg_signals" in results
    assert "landmarks" in results
    assert "pose_angles" in results
    
    face_crops = results["face_crops"]
    residual_maps = results["residual_maps"]
    rppg_signals = results["rppg_signals"]
    landmarks = results["landmarks"]
    pose_angles = results["pose_angles"]
    
    # Check shapes
    assert face_crops.shape == (num_frames, 224, 224, 3)
    assert residual_maps.shape == (num_frames, 224, 224, 1)
    assert rppg_signals.shape == (num_frames, 3) # R, G, B channels
    assert landmarks.shape == (num_frames, 68, 2) # 68-point (x, y) landmarks
    assert pose_angles.shape == (num_frames, 3) # pitch, yaw, roll
    
    # Check that frame 3 (index 3) is a zero-tensor because of NaN bbox
    assert np.all(face_crops[3] == 0)
    assert np.all(landmarks[3] == 0) # NaN bbox -> zero landmarks
    
    # Verify smoothing didn't corrupt dimensionality
    assert not np.any(np.isnan(rppg_signals[0]))
