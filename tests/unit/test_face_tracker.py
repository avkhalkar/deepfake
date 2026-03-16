"""
Unit tests for the FaceTracker component.

This suite uses a mocked MTCNN to verify the "First-Select, Then-Lock"
tracking logic, including temporal IOU matching and Kalman Filter
predictions for occluded frames, to ensure the primary subject's tracked
identity persists across the video without jumping between faces.

Usage:
    pytest tests/unit/test_face_tracker.py -v
"""

import numpy as np
import pytest
from src.face_tracker import FaceTracker, iou

class MockMTCNN:
    """Mock for MTCNN to provide predictable bounding boxes without NN overhead"""
    def __init__(self, keep_all=True, device='cpu'):
        self.frame_idx = 0
        
    def detect(self, frame):
        # Frame 0: One large face, one small face
        # Frame 1: Same faces moved slightly
        # Frame 2: Top face missing (occluded), small face remains
        # Frame 3: Everything returns
        
        if self.frame_idx == 0:
            boxes = np.array([[10, 10, 110, 110], [200, 200, 220, 220]]) # Large is [10,10,110,110]
        elif self.frame_idx == 1:
            boxes = np.array([[15, 15, 115, 115], [205, 205, 225, 225]])
        elif self.frame_idx == 2:
            boxes = np.array([[210, 210, 230, 230]]) # Primary is occluded
        elif self.frame_idx == 3:
            boxes = np.array([[20, 20, 120, 120], [215, 215, 235, 235]])
        else:
            boxes = np.empty((0, 4))
            
        self.frame_idx += 1
        return boxes, None

def test_iou():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([5, 5, 15, 15])
    
    # intersection is 5x5 = 25
    # union is 100 + 100 - 25 = 175
    # IOU = 25/175 = 1/7 ~= 0.1428
    iou_val = iou(box1, box2)
    assert np.isclose(iou_val, 1/7)

def test_face_tracker_identity_locking(monkeypatch):
    """
    Tests the "First-Select, Then-Lock" tracking logic using a mocked MTCNN
    """
    tracker = FaceTracker(device='cpu')
    tracker.mtcnn = MockMTCNN() # Inject mock
    
    # Create 4 dummy frames
    frames = [np.zeros((300, 300, 3)) for _ in range(4)]
    
    results = tracker.process_frames(frames)
    boxes = results["primary_subject_boxes"]
    
    # Check that 4 boxes were returned
    assert len(boxes) == 4
    
    # Frame 0: Primary track should be the largest one: [10, 10, 110, 110]
    assert np.allclose(boxes[0], [10, 10, 110, 110], atol=1.0)
    
    # Frame 1: Primary track should follow the movement: [15, 15, 115, 115]
    assert np.allclose(boxes[1], [15, 15, 115, 115], atol=1.0)
    
    # Frame 2: Primary is occluded, so Kalman Filter should predict its next location (~[20, 20, 120, 120])
    # The output from the tracker will be the prediction, it might not be perfect, 
    # but it shouldn't jump to the small face [210, 210, 230, 230]
    predicted = boxes[2]
    assert not np.isnan(predicted[0])
    assert predicted[0] < 100 # X coordinate should be small, not ~200
    
    # Frame 3: Primary returns, it should successfully match again
    assert np.allclose(boxes[3], [20, 20, 120, 120], atol=5.0)

