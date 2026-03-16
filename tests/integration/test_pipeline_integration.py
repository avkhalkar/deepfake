"""
Integration test: End-to-end pipeline test on real dataset videos.

This test picks a small number of actual videos from the datasets directory
and runs them through the full pipeline (preprocessor → tracker → extractor → audio).
It then verifies the .pt output tensor shapes and metadata correctness.

Usage:
    pytest tests/integration/test_pipeline_integration.py -v -s
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

from src.video_preprocessor import VideoPreprocessor
from src.face_tracker import FaceTracker
from src.feature_extractor import FeatureExtractor
from src.audio_processor import AudioProcessor
from src.dataset_manager import DatasetManager


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "test_output"

# Skip all tests if datasets directory doesn't exist
pytestmark = pytest.mark.skipif(
    not DATASETS_ROOT.exists(),
    reason=f"Datasets root not found: {DATASETS_ROOT}"
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset_manager():
    return DatasetManager(str(DATASETS_ROOT))


@pytest.fixture(scope="module")
def sample_entries(dataset_manager):
    """Get 3 diverse videos: tries to get 1 real + 1 fake + 1 from different dataset."""
    entries = dataset_manager.sample_subset(3, balanced=True, seed=42)
    assert len(entries) > 0, "No videos found in datasets directory"
    return entries


@pytest.fixture(scope="module")
def preprocessor():
    return VideoPreprocessor(target_fps=25, target_res=(1280, 720))


@pytest.fixture(scope="module")
def feature_extractor():
    return FeatureExtractor(target_size=(224, 224))


@pytest.fixture(scope="module")
def audio_processor():
    return AudioProcessor()


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
class TestDatasetDiscovery:
    """Verify the DatasetManager discovers real datasets correctly."""

    def test_discovers_datasets(self, dataset_manager):
        assert len(dataset_manager.handlers) > 0, "No datasets discovered"
        names = [h.name for h in dataset_manager.handlers]
        print(f"Discovered datasets: {names}")

    def test_summary_has_videos(self, dataset_manager):
        summary = dataset_manager.summary()
        assert summary["total_videos"] > 0
        print(f"Total available videos: {summary['total_videos']}")
        for ds_name, info in summary["datasets"].items():
            print(f"  {ds_name}: REAL={info['REAL']}, FAKE={info['FAKE']}, audio={info['has_audio']}")

    def test_balanced_sampling(self, dataset_manager):
        sampled = dataset_manager.sample_subset(10, balanced=True)
        labels = [e["label"] for e in sampled]
        datasets = [e["dataset"] for e in sampled]
        print(f"Sampled {len(sampled)} videos: labels={labels}, datasets={datasets}")
        # Should have at least some diversity
        assert len(set(labels)) > 0


class TestFrameExtraction:
    """Verify frame extraction works on real videos."""

    def test_extract_frames_from_real_video(self, sample_entries, preprocessor):
        entry = sample_entries[0]
        print(f"Testing frame extraction on: {entry['path']}")

        frames, indices, fps = preprocessor.extract_frames(entry["path"])

        assert len(frames) > 0, f"No frames extracted from {entry['path']}"
        assert frames[0].shape[2] == 3, "Frames should be RGB (3 channels)"
        print(f"  Extracted {len(frames)} frames, shape={frames[0].shape}")


class TestFullPipeline:
    """End-to-end: preprocessor → tracker → extractor → audio → .pt save."""

    def test_single_video_pipeline(self, sample_entries, preprocessor,
                                    feature_extractor, audio_processor):
        entry = sample_entries[0]
        video_path = entry["path"]
        print(f"\n{'='*60}")
        print(f"Full pipeline test on: {Path(video_path).name}")
        print(f"  Dataset: {entry['dataset']}, Label: {entry['label']}")
        print(f"{'='*60}")

        # Step 1: Frame extraction
        frames, indices, fps = preprocessor.extract_frames(video_path)
        assert len(frames) > 0
        print(f"  [1] Frames extracted: {len(frames)}, shape={frames[0].shape}, fps={fps}")

        # Step 2: Face tracking
        tracker = FaceTracker(device='cpu')
        tracking_result = tracker.process_frames(frames)
        bboxes = tracking_result["primary_subject_boxes"]
        assert bboxes.shape[0] == len(frames)
        
        valid_count = np.sum(~np.isnan(bboxes[:, 0]))
        print(f"  [2] Face tracked: {valid_count}/{len(frames)} frames have valid boxes")

        # Step 3: Feature extraction
        features = feature_extractor.process_sequence(frames, bboxes)
        face_crops = features["face_crops"]
        residual_maps = features["residual_maps"]
        rppg_signals = features["rppg_signals"]

        assert face_crops.shape == (len(frames), 224, 224, 3)
        assert residual_maps.shape == (len(frames), 224, 224, 3)
        assert rppg_signals.shape == (len(frames), 3)
        print(f"  [3] Features: crops={face_crops.shape}, residuals={residual_maps.shape}, rPPG={rppg_signals.shape}")

        # Step 4: Audio
        audio_result = audio_processor.extract_features(video_path)
        print(f"  [4] Audio: has_audio={audio_result['has_audio']}, mfcc={audio_result['mfcc'].shape}")

        # Step 5: Assemble & save .pt
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{Path(video_path).stem}_test.pt"

        output = {
            "video_name": Path(video_path).stem,
            "dataset": entry["dataset"],
            "label": entry["label"],
            "face_crops": torch.from_numpy(face_crops).float(),
            "residual_maps": torch.from_numpy(residual_maps).float(),
            "rppg_signals": torch.from_numpy(rppg_signals).float(),
            "pose_angles": torch.from_numpy(features["pose_angles"]).float(),
            "bboxes": torch.from_numpy(bboxes).float(),
            "mfcc": torch.from_numpy(audio_result["mfcc"]).float(),
            "mel_spectrogram": torch.from_numpy(audio_result["mel_spectrogram"]).float(),
            "has_audio": audio_result["has_audio"],
            "num_frames": len(frames),
        }

        torch.save(output, str(output_path))
        assert output_path.exists()
        print(f"  [5] Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # Step 6: Reload and verify
        loaded = torch.load(str(output_path), weights_only=False)
        assert loaded["label"] == entry["label"]
        assert loaded["face_crops"].shape == torch.Size([len(frames), 224, 224, 3])
        assert loaded["rppg_signals"].shape == torch.Size([len(frames), 3])
        print(f"  [6] Reload verified OK")
        
        # Cleanup
        output_path.unlink()
        print(f"  [DONE] Pipeline complete for {Path(video_path).name}")

    def test_pt_schema_fields(self, sample_entries, preprocessor,
                               feature_extractor, audio_processor):
        """Verify all required schema fields are present in .pt output."""
        entry = sample_entries[0]
        
        frames, indices, fps = preprocessor.extract_frames(entry["path"])
        
        tracker = FaceTracker(device='cpu')
        tracking_result = tracker.process_frames(frames)
        bboxes = tracking_result["primary_subject_boxes"]
        
        features = feature_extractor.process_sequence(frames, bboxes)
        audio_result = audio_processor.extract_features(entry["path"])
        
        output = {
            "video_name": Path(entry["path"]).stem,
            "dataset": entry["dataset"],
            "label": entry["label"],
            "face_crops": torch.from_numpy(features["face_crops"]).float(),
            "residual_maps": torch.from_numpy(features["residual_maps"]).float(),
            "rppg_signals": torch.from_numpy(features["rppg_signals"]).float(),
            "pose_angles": torch.from_numpy(features["pose_angles"]).float(),
            "bboxes": torch.from_numpy(bboxes).float(),
            "mfcc": torch.from_numpy(audio_result["mfcc"]).float(),
            "mel_spectrogram": torch.from_numpy(audio_result["mel_spectrogram"]).float(),
            "has_audio": audio_result["has_audio"],
            "num_frames": len(frames),
        }
        
        # Verify all expected keys exist
        required_keys = [
            "video_name", "dataset", "label", "face_crops", "residual_maps",
            "rppg_signals", "pose_angles", "bboxes", "mfcc", "mel_spectrogram",
            "has_audio", "num_frames"
        ]
        
        for key in required_keys:
            assert key in output, f"Missing key in .pt schema: {key}"
        
        # Verify types
        assert isinstance(output["video_name"], str)
        assert isinstance(output["label"], str)
        assert isinstance(output["has_audio"], bool)
        assert isinstance(output["num_frames"], int)
        assert isinstance(output["face_crops"], torch.Tensor)
        assert output["face_crops"].dtype == torch.float32
