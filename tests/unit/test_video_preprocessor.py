"""
Unit tests for the VideoPreprocessor component.

This suite verifies that the preprocessor correctly reads video files,
extracts frames at the specified sequence length and FPS, and resizes
them to the target resolution.

Usage:
    pytest tests/unit/test_video_preprocessor.py -v
"""

import os
import cv2
import numpy as np
import pytest
import tempfile
from src.video_preprocessor import VideoPreprocessor

@pytest.fixture
def dummy_video_file():
    """Creates a temporary 1-second 50 FPS dummy video for testing."""
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    
    # Video properties
    width, height = 800, 600
    fps = 50
    duration_sec = 1
    total_frames = fps * duration_sec
    
    # Write a dummy video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for i in range(total_frames):
        # Create a simple frame consisting of random noise
        # This prevents codecs from heavily compressing zero-frames causing issues
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
        
    out.release()
    
    yield temp_path
    
    # Cleanup after test
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_video_preprocessor_initialization():
    target_fps = 10
    target_res = (640, 480)
    preprocessor = VideoPreprocessor(target_fps=target_fps, target_res=target_res)
    assert preprocessor.target_fps == target_fps
    assert preprocessor.target_res == target_res

def test_extract_frames_shape_and_fps(dummy_video_file):
    # The dummy video is 50 FPS, 1 second long (50 frames), 800x600 resolution
    # We want to extract at 25 FPS, 1280x720 resolution
    target_fps = 25
    target_res = (1280, 720)
    
    preprocessor = VideoPreprocessor(target_fps=target_fps, target_res=target_res)
    frames, indices, source_fps = preprocessor.extract_frames(dummy_video_file)
    
    # Check source FPS is correctly identified
    assert source_fps == 50.0
    
    # Since duration is 1 second, and target_fps is 25, we should get exactly 25 frames
    assert len(frames) == 25
    assert len(indices) == 25
    
    # Check shape of the returned tensor/array: [T, H, W, C]
    # Note: OpenCV target_res is (Width, Height), but numpy shape is (Height, Width)
    expected_height, expected_width = target_res[1], target_res[0]
    assert frames.shape == (25, expected_height, expected_width, 3)
    
    # Ensure it's in uint8 format (since color values are 0-255)
    assert frames.dtype == np.uint8

def test_extract_frames_file_not_found():
    preprocessor = VideoPreprocessor()
    with pytest.raises(FileNotFoundError):
        preprocessor.extract_frames("non_existent_video_path.mp4")
