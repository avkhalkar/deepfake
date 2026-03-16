"""
dataset_manager.py
------------------
Provides a unified view over four deepfake-detection datasets:

    Dataset            | Root folder          | Label source         | Audio?
    -------------------|----------------------|----------------------|-------
    Celeb-DF-v2        | Celeb-DF-v2/         | directory names      | No
    DFDC               | dfdc/                | metadata.json        | Yes
    DeeperForensics    | deeperforensics/     | directory names      | No
    FaceForensics++    | face_forensics/      | directory names      | No

Each concrete handler inherits from ``BaseDataset`` and implements
``get_video_entries()`` which returns a flat list of dicts:

    {"path": str, "label": "REAL"|"FAKE", "dataset": str, "has_audio": bool}

``DatasetManager`` auto-discovers whichever datasets are present under a
root directory and exposes filtering and balanced-sampling utilities.

Usage:
    from dataset_manager import DatasetManager

    dm = DatasetManager("path/to/datasets")

    dm.summary()                                # quick overview
    dm.get_all_entries()                         # every video across all datasets
    dm.filter_by_label("FAKE")                   # only fake videos
    dm.filter_by_dataset("DFDC")                 # only DFDC
    dm.sample_subset(100, balanced=True)         # 100 videos, balanced across datasets

    # Or use a single handler directly:
    from dataset_manager import CelebDFDataset
    ds = CelebDFDataset("path/to/Celeb-DF-v2")
    entries = ds.get_video_entries()
"""

import os
import json
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional


# ======================================================================
#  Base class
# ======================================================================
class BaseDataset(ABC):
    """
    Abstract base for all dataset handlers.

    Subclasses must implement ``get_video_entries()`` which scans the
    on-disk directory and returns a standardised list of video entries.
    """

    def __init__(self, root_dir: str, name: str):
        self.root_dir = Path(root_dir)
        self.name = name
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    @abstractmethod
    def get_video_entries(self) -> List[Dict]:
        """
        Return a list of dicts with keys:
            path              – absolute path to the .mp4 file
            label             – "REAL" | "FAKE"
            dataset           – human-readable dataset name
            has_audio         – whether the dataset carries audio
            manipulation_type – str or None (e.g., "Deepfakes", "FaceSwap")
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(root='{self.root_dir}')"


# ======================================================================
#  Celeb-DF-v2
# ======================================================================
class CelebDFDataset(BaseDataset):
    """
    Handler for the Celeb-DF-v2 dataset.

    Directory layout:
        Celeb-DF-v2/
        |-- Celeb-real/         -> REAL
        |-- Celeb-synthesis/    -> FAKE
        |-- YouTube-real/       -> REAL
        +-- List_of_testing_videos.txt
    """

    def __init__(self, root_dir: str):
        super().__init__(root_dir, "Celeb-DF-v2")

    def get_video_entries(self) -> List[Dict]:
        entries = []
        # folder name -> label
        label_map = {
            "Celeb-real":      "REAL",
            "YouTube-real":    "REAL",
            "Celeb-synthesis": "FAKE",
        }
        for folder, label in label_map.items():
            folder_path = self.root_dir / folder
            if not folder_path.exists():
                continue
            for mp4 in sorted(folder_path.glob("*.mp4")):
                entries.append({
                    "path": str(mp4), "label": label,
                    "dataset": self.name, "has_audio": False,
                    "manipulation_type": None,
                })
        return entries


# ======================================================================
#  DFDC (DeepFake Detection Challenge)
# ======================================================================
class DFDCDataset(BaseDataset):
    """
    Handler for Facebook's DFDC dataset.

    Directory layout:
        dfdc/
        +-- train_sample_videos/
            |-- metadata.json      {"file.mp4": {"label": "FAKE", ...}}
            +-- *.mp4

    Only dataset with audio.
    """

    def __init__(self, root_dir: str):
        super().__init__(root_dir, "DFDC")

    def get_video_entries(self) -> List[Dict]:
        entries = []
        videos_dir    = self.root_dir / "train_sample_videos"
        metadata_path = videos_dir / "metadata.json"

        if not metadata_path.exists():
            # fallback: scan directory without metadata (label = UNKNOWN)
            for mp4 in sorted(videos_dir.glob("*.mp4")):
                entries.append({
                    "path": str(mp4), "label": "UNKNOWN",
                    "dataset": self.name, "has_audio": True,
                    "manipulation_type": None,
                })
            return entries

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        for filename, info in sorted(metadata.items()):
            video_path = videos_dir / filename
            if video_path.exists():
                entries.append({
                    "path": str(video_path),
                    "label": info.get("label", "UNKNOWN"),
                    "dataset": self.name,
                    "has_audio": True,
                    "manipulation_type": None,
                })
        return entries


# ======================================================================
#  DeeperForensics-1.0
# ======================================================================
class DeeperForensicsDataset(BaseDataset):
    """
    Handler for DeeperForensics-1.0.

    Directory layout:
        deeperforensics/
        |-- source_videos_part_03/        -> REAL  (flat .mp4)
        |-- manipulated_videos_part_01/   -> FAKE  (flat .mp4)
        +-- lists/                        (text manifests, not used here)
    """

    def __init__(self, root_dir: str):
        super().__init__(root_dir, "DeeperForensics")

    def get_video_entries(self) -> List[Dict]:
        entries = []
        dir_label = {
            "source_videos_part_03":      "REAL",
            "manipulated_videos_part_01": "FAKE",
        }
        for folder, label in dir_label.items():
            folder_path = self.root_dir / folder
            if not folder_path.exists():
                continue
            for mp4 in sorted(folder_path.rglob("*.mp4")):
                entries.append({
                    "path": str(mp4), "label": label,
                    "dataset": self.name, "has_audio": False,
                    "manipulation_type": None,
                })
        return entries


# ======================================================================
#  FaceForensics++
# ======================================================================
class FaceForensicsDataset(BaseDataset):
    """
    Handler for FaceForensics++.

    Directory layout:
        face_forensics/
        |-- original_sequences/    -> REAL
        |-- Deepfakes/             -> FAKE
        |-- Face2Face/             -> FAKE
        |-- FaceShifter/           -> FAKE
        |-- FaceSwap/              -> FAKE
        |-- NeuralTextures/        -> FAKE
        +-- csv/                   (metadata CSVs, not used here)
    """

    def __init__(self, root_dir: str):
        super().__init__(root_dir, "FaceForensics++")

    def get_video_entries(self) -> List[Dict]:
        entries = []

        real_dirs = ["original-sequences-c23-videos"]
        fake_dirs = [
             "manipulated-sequences-Deepfakes-c23-videos",
             "manipulated-sequences-Face2Face-c23-videos",
             "manipulated-sequences-FaceShifter-c23-videos",
             "manipulated-sequences-FaceSwap-c23-videos",
             "manipulated-sequences-NeuralTextures-c23-videos"
        ]

        for folder in real_dirs:
            path = self.root_dir / folder
            if not path.exists():
                continue
            for mp4 in sorted(path.glob("*.mp4")):
                entries.append({
                    "path": str(mp4), "label": "REAL",
                    "dataset": self.name, "has_audio": False,
                    "manipulation_type": None,
                })

        for folder in fake_dirs:
            path = self.root_dir / folder
            if not path.exists():
                continue
            # Extract manipulation type from directory name
            # e.g. "manipulated-sequences-FaceSwap-c23-videos" -> "FaceSwap"
            parts = folder.split("-")
            manip_type = parts[2] if len(parts) >= 3 else None
            for mp4 in sorted(path.glob("*.mp4")):
                entries.append({
                    "path": str(mp4), "label": "FAKE",
                    "dataset": self.name, "has_audio": False,
                    "manipulation_type": manip_type,
                })
        return entries


# ======================================================================
#  DatasetManager  –  unified aggregation layer
# ======================================================================
class DatasetManager:
    """
    Auto-discovers datasets under a root directory and provides
    filtering, sampling, and summary utilities.

    Parameters
    ----------
    datasets_root : str
        Parent directory that contains one or more dataset folders
        (``Celeb-DF-v2/``, ``dfdc/``, ``deeperforensics/``, ``face_forensics/``).
    """

    def __init__(self, datasets_root: str):
        self.datasets_root = Path(datasets_root)
        self.handlers: List[BaseDataset] = []
        self._entries: Optional[List[Dict]] = None
        self._auto_discover()

    # ------------------------------------------------------------------
    #  Discovery
    # ------------------------------------------------------------------
    def _auto_discover(self):
        """Scan *datasets_root* for known dataset folders."""
        dataset_map = {
            "Celeb-DF-v2":    CelebDFDataset,
            "dfdc":           DFDCDataset,
            "deeperforensics": DeeperForensicsDataset,
            "face_forensics": FaceForensicsDataset,
        }
        for folder_name, cls in dataset_map.items():
            path = self.datasets_root / folder_name
            if path.exists():
                try:
                    self.handlers.append(cls(str(path)))
                except FileNotFoundError:
                    pass

    # ------------------------------------------------------------------
    #  Retrieval
    # ------------------------------------------------------------------
    def get_all_entries(self) -> List[Dict]:
        """Return every video entry from every discovered dataset (cached)."""
        if self._entries is None:
            self._entries = []
            for handler in self.handlers:
                self._entries.extend(handler.get_video_entries())
        return self._entries

    def filter_by_label(self, label: str) -> List[Dict]:
        """Return only entries whose label matches *label* ('REAL' or 'FAKE')."""
        return [e for e in self.get_all_entries() if e["label"] == label]

    def filter_by_dataset(self, dataset_name: str) -> List[Dict]:
        """Return only entries from a specific dataset."""
        return [e for e in self.get_all_entries() if e["dataset"] == dataset_name]

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------
    def sample_subset(self, n: int, balanced: bool = True, seed: int = 42) -> List[Dict]:
        """
        Draw *n* videos. When ``balanced=True``, guarantees exact hierarchical 
        allocation across datasets and then labels.
        """
        import random
        rng = random.Random(seed)

        all_entries = self.get_all_entries()
        if not balanced or n >= len(all_entries):
            rng.shuffle(all_entries)
            return all_entries[:n]

        # 1. Group by dataset first
        datasets: Dict[str, list] = {}
        for entry in all_entries:
            datasets.setdefault(entry["dataset"], []).append(entry)

        dataset_names = list(datasets.keys())
        rng.shuffle(dataset_names)
        
        # 2. Allocate exact number of videos per dataset
        allocations = {name: n // len(dataset_names) for name in dataset_names}
        for name in dataset_names[:n % len(dataset_names)]:
            allocations[name] += 1
            
        sampled = []
        for ds_name, target_count in allocations.items():
            if target_count == 0:
                continue
            
            # Group by label within this dataset
            labels_group = {"REAL": [], "FAKE": [], "UNKNOWN": []}
            for e in datasets[ds_name]:
                labels_group[e["label"]].append(e)
            
            labels_group = {k: v for k, v in labels_group.items() if v}
            label_names = list(labels_group.keys())
            rng.shuffle(label_names)
            
            # 3. Allocate exact number of videos per label
            lbl_alloc = {lname: target_count // len(label_names) for lname in label_names}
            for lname in label_names[:target_count % len(label_names)]:
                lbl_alloc[lname] += 1
                
            for lname, count in lbl_alloc.items():
                if count > 0:
                    entries = labels_group[lname]
                    rng.shuffle(entries)
                    sampled.extend(entries[:count])

        rng.shuffle(sampled)
        return sampled[:n]

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        """Return a nested dict summarising video counts per dataset."""
        all_entries = self.get_all_entries()
        result = {"total_videos": len(all_entries), "datasets": {}}

        for entry in all_entries:
            name = entry["dataset"]
            if name not in result["datasets"]:
                result["datasets"][name] = {
                    "REAL": 0, "FAKE": 0, "UNKNOWN": 0,
                    "has_audio": entry["has_audio"],
                }
            label = entry["label"]
            if label in result["datasets"][name]:
                result["datasets"][name][label] += 1
            else:
                result["datasets"][name]["UNKNOWN"] += 1

        return result
