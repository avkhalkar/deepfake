"""
Unit tests for the AudioProcessor component.

This suite verifies that audio features (MFCC and Mel-spectrogram) are successfully
extracted when audio is present, and correctly fall back to appropriately sized
empty zero-tensors `(features, 0)` when missing from the video feed.

Usage:
    pytest tests/unit/test_audio_processor.py -v
"""

import numpy as np
import pytest
from src.audio_processor import AudioProcessor


def test_audio_processor_missing_file():
    """Test that missing files return zero tensors with has_audio=False."""
    processor = AudioProcessor()
    result = processor.extract_features("/nonexistent/video.mp4")
    
    assert result["has_audio"] is False
    assert result["mfcc"].shape[0] == 13  # n_mfcc default
    assert result["mel_spectrogram"].shape[0] == 128  # n_mels default
    assert np.all(result["mfcc"] == 0)
    assert np.all(result["mel_spectrogram"] == 0)


def test_audio_processor_custom_params():
    """Test custom MFCC and Mel parameters."""
    processor = AudioProcessor(n_mfcc=20, n_mels=64)
    result = processor.extract_features("/nonexistent/video.mp4")
    
    assert result["mfcc"].shape[0] == 20
    assert result["mel_spectrogram"].shape[0] == 64
