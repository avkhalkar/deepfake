"""
Integration test: Verifies the resulting .pt schema tensors.

This test automatically scans the `test_output/` directory for any generated
`.pt` payload files and rigorously validates that all keys, types, values,
and spatial/temporal tensor shapes exactly conform to the Phase 1 schema.

Usage:
    pytest tests/integration/test_output_files.py -v
"""

import pytest
import torch
from pathlib import Path

# The directory where the pipeline saves its output .pt files
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "test_output"

# Skip tests if no output directory exists
pytestmark = pytest.mark.skipif(
    not OUTPUT_DIR.exists(),
    reason=f"Output directory not found: {OUTPUT_DIR}"
)

def get_all_pt_files():
    """Discover all .pt files generated in the test_output directory."""
    if not OUTPUT_DIR.exists():
        return []
    return list(OUTPUT_DIR.rglob("*.pt"))

@pytest.fixture(params=get_all_pt_files())
def pt_file(request):
    """Fixture that parameterizes a test for every .pt file found."""
    return request.param

def test_pt_file_schema(pt_file: Path):
    """
    Loads a .pt file and perfectly verifies its schema against the Phase 1 specification.
    """
    print(f"\nVerifying schema for: {pt_file.relative_to(OUTPUT_DIR)}")
    
    # Load the torch dict
    # We use weights_only=False because we are loading dictionary objects, not just model state dicts.
    try:
        data = torch.load(str(pt_file), weights_only=False)
    except EOFError:
        pytest.fail(f"Corrupted or empty .pt file: {pt_file}")
        
    # 1. Verify all expected keys are present
    expected_keys = {
        "video_name", "dataset", "label", "has_audio", "num_frames",
        "manipulation_type", "landmarks",
        "face_crops", "residual_maps", "rppg_signals", "pose_angles", "bboxes",
        "mfcc", "mel_spectrogram"
    }
    
    missing_keys = expected_keys - set(data.keys())
    assert not missing_keys, f"File {pt_file.name} is missing keys: {missing_keys}"
    
    # 2. Verify basic metadata types and values
    assert isinstance(data["video_name"], str)
    assert isinstance(data["dataset"], str)
    
    assert data["label"] in ["REAL", "FAKE", "UNKNOWN"], f"Invalid label: {data['label']}"
    assert isinstance(data["has_audio"], bool)
    
    T = data["num_frames"]
    assert isinstance(T, int)
    assert T > 0, "num_frames must be > 0"
    
    # 3. Verify spatial/temporal tensor shapes
    # All visual tensors should have their first dimension equal to T (num_frames tracked)
    assert data["face_crops"].shape == (T, 224, 224, 3), f"face_crops shape mismatch: {data['face_crops'].shape}"
    assert data["residual_maps"].shape == (T, 224, 224, 1), f"residual_maps is not [T, 224, 224, 1], got {data['residual_maps'].shape}"
    assert data["rppg_signals"].shape == (T, 3), f"rppg_signals shape mismatch: {data['rppg_signals'].shape}"
    
    # Pose angles are real values from solvePnP
    assert data["pose_angles"].shape == (T, 3)
    
    # 68-point landmarks
    assert data["landmarks"].shape == (T, 68, 2), f"landmarks shape mismatch: {data['landmarks'].shape}"
    
    # Bounding boxes are [T, 4]
    assert data["bboxes"].shape == (T, 4)
    
    # Manipulation type should be string or None
    manip = data.get("manipulation_type")
    assert manip is None or isinstance(manip, str), f"manipulation_type must be str or None, got {type(manip)}"
    
    # 4. Verify Audio Tensors
    if data["has_audio"]:
        # If true, the tensor should NOT be strictly empty. Shape should be [features, time]
        assert data["mfcc"].numel() > 0, "has_audio is True but mfcc is empty"
        assert data["mel_spectrogram"].numel() > 0, "has_audio is True but mel_spectrogram is empty"
        
        # Audio feature dimensions
        assert data["mfcc"].shape[0] == 13, f"Expected 13 MFCC coefficients, got {data['mfcc'].shape}"
        assert data["mel_spectrogram"].shape[0] == 128, f"Expected 128 mel bands, got {data['mel_spectrogram'].shape}"
    else:
        # If no audio, arrays should be identically empty Shape=[0]
        assert data["mfcc"].numel() == 0, f"has_audio is False but mfcc has shape {data['mfcc'].shape}"
        assert data["mel_spectrogram"].numel() == 0, f"has_audio is False but mel_spectrogram has shape {data['mel_spectrogram'].shape}"
    
    # 5. Verify Tensor Data Types
    expected_dtype = torch.float32
    for key in ["face_crops", "residual_maps", "rppg_signals", "pose_angles", "bboxes", "landmarks", "mfcc", "mel_spectrogram"]:
        if data[key].numel() > 0:  # Only check dtype if tensor is not empty
            assert data[key].dtype == expected_dtype, f"{key} tensor must be float32, got {data[key].dtype}"

    print(f"  [OK] Schema fully verified for {pt_file.name}")
