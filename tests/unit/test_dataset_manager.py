"""
Unit tests for the DatasetManager components.

This suite handles mock filesystem creation to thoroughly test automatic dataset
discovery, hierarchical subset sampling (by dataset and label allocations), 
and dataset-specific handler behavior (`DeeperForensics`, `DFDC`, etc.).

Usage:
    pytest tests/unit/test_dataset_manager.py -v
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from src.dataset_manager import (
    BaseDataset, CelebDFDataset, DFDCDataset,
    DeeperForensicsDataset, FaceForensicsDataset, DatasetManager
)


@pytest.fixture
def mock_celeb_df(tmp_path):
    """Create a mock Celeb-DF-v2 directory structure."""
    ds_root = tmp_path / "Celeb-DF-v2"
    
    (ds_root / "Celeb-real").mkdir(parents=True)
    (ds_root / "Celeb-synthesis").mkdir(parents=True)
    (ds_root / "YouTube-real").mkdir(parents=True)
    
    # Create dummy mp4 files  
    for i in range(3):
        (ds_root / "Celeb-real" / f"real_{i:03d}.mp4").touch()
        (ds_root / "Celeb-synthesis" / f"fake_{i:03d}.mp4").touch()
    (ds_root / "YouTube-real" / "yt_001.mp4").touch()
    
    return str(ds_root)


@pytest.fixture
def mock_dfdc(tmp_path):
    """Create a mock DFDC directory structure."""
    ds_root = tmp_path / "dfdc"
    videos_dir = ds_root / "train_sample_videos"
    videos_dir.mkdir(parents=True)
    
    metadata = {
        "real_001.mp4": {"label": "REAL", "split": "train", "original": None},
        "fake_001.mp4": {"label": "FAKE", "split": "train", "original": "real_001.mp4"},
        "fake_002.mp4": {"label": "FAKE", "split": "train", "original": "real_001.mp4"},
    }
    
    for filename in metadata:
        (videos_dir / filename).touch()
    
    with open(videos_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    return str(ds_root)


@pytest.fixture
def mock_deeper(tmp_path):
    """Create a mock DeeperForensics directory structure."""
    ds_root = tmp_path / "deeperforensics"
    (ds_root / "source_videos_part_03").mkdir(parents=True)
    (ds_root / "manipulated_videos_part_01").mkdir(parents=True)
    
    for i in range(2):
        (ds_root / "source_videos_part_03" / f"src_{i:03d}.mp4").touch()
        (ds_root / "manipulated_videos_part_01" / f"manip_{i:03d}.mp4").touch()
    
    return str(ds_root)


@pytest.fixture
def mock_ff(tmp_path):
    """Create a mock FaceForensics++ directory structure."""
    ds_root = tmp_path / "face_forensics"
    (ds_root / "original-sequences-c23-videos").mkdir(parents=True)
    (ds_root / "manipulated-sequences-Deepfakes-c23-videos").mkdir(parents=True)
    (ds_root / "manipulated-sequences-Face2Face-c23-videos").mkdir(parents=True)
    
    (ds_root / "original-sequences-c23-videos" / "orig_001.mp4").touch()
    (ds_root / "manipulated-sequences-Deepfakes-c23-videos" / "df_001.mp4").touch()
    (ds_root / "manipulated-sequences-Face2Face-c23-videos" / "f2f_001.mp4").touch()
    
    return str(ds_root)


def test_celeb_df_dataset(mock_celeb_df):
    ds = CelebDFDataset(mock_celeb_df)
    entries = ds.get_video_entries()
    
    # 3 real + 3 fake + 1 youtube real = 7
    assert len(entries) == 7
    
    real_count = sum(1 for e in entries if e["label"] == "REAL")
    fake_count = sum(1 for e in entries if e["label"] == "FAKE")
    assert real_count == 4  # 3 Celeb-real + 1 YouTube-real
    assert fake_count == 3
    
    # No audio in Celeb-DF
    assert all(e["has_audio"] is False for e in entries)


def test_dfdc_dataset(mock_dfdc):
    ds = DFDCDataset(mock_dfdc)
    entries = ds.get_video_entries()
    
    assert len(entries) == 3
    assert sum(1 for e in entries if e["label"] == "REAL") == 1
    assert sum(1 for e in entries if e["label"] == "FAKE") == 2
    
    # DFDC always has audio
    assert all(e["has_audio"] is True for e in entries)


def test_deeper_forensics_dataset(mock_deeper):
    ds = DeeperForensicsDataset(mock_deeper)
    entries = ds.get_video_entries()
    
    assert len(entries) == 4
    assert sum(1 for e in entries if e["label"] == "REAL") == 2
    assert sum(1 for e in entries if e["label"] == "FAKE") == 2


def test_face_forensics_dataset(mock_ff):
    ds = FaceForensicsDataset(mock_ff)
    entries = ds.get_video_entries()
    
    assert len(entries) == 3
    assert sum(1 for e in entries if e["label"] == "REAL") == 1
    assert sum(1 for e in entries if e["label"] == "FAKE") == 2


def test_dataset_manager_auto_discovery(tmp_path):
    """Test that DatasetManager discovers datasets automatically."""
    # Create all four mock datasets under one root
    datasets_root = tmp_path / "datasets"

    celeb = datasets_root / "Celeb-DF-v2" / "Celeb-real"
    celeb.mkdir(parents=True)
    (celeb / "test.mp4").touch()

    dfdc = datasets_root / "dfdc" / "train_sample_videos"
    dfdc.mkdir(parents=True)
    (dfdc / "test.mp4").touch()
    with open(dfdc / "metadata.json", 'w') as f:
        json.dump({"test.mp4": {"label": "REAL", "split": "train", "original": None}}, f)

    manager = DatasetManager(str(datasets_root))
    
    assert len(manager.handlers) >= 2  # At least Celeb-DF and DFDC

    all_entries = manager.get_all_entries()
    assert len(all_entries) >= 2


def test_dataset_manager_sampling(tmp_path):
    """Test balanced sampling across datasets."""
    datasets_root = tmp_path / "datasets"
    
    # Create Celeb-DF with 10 videos
    celeb_real = datasets_root / "Celeb-DF-v2" / "Celeb-real"
    celeb_fake = datasets_root / "Celeb-DF-v2" / "Celeb-synthesis"
    celeb_real.mkdir(parents=True)
    celeb_fake.mkdir(parents=True)
    for i in range(5):
        (celeb_real / f"real_{i:03d}.mp4").touch()
        (celeb_fake / f"fake_{i:03d}.mp4").touch()

    manager = DatasetManager(str(datasets_root))

    # Sample 4 videos  
    sampled = manager.sample_subset(4, balanced=True)
    assert len(sampled) == 4


def test_base_dataset_not_found():
    with pytest.raises(FileNotFoundError):
        CelebDFDataset("/nonexistent/path/to/dataset")
