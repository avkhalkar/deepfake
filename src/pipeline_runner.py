"""
pipeline_runner.py
------------------
Orchestrates the full Phase 1 data-extraction pipeline, wiring together:

    VideoPreprocessor  ->  FaceTracker  ->  FeatureExtractor  ->  AudioProcessor

Each video is processed into a single ``.pt`` file containing all extracted
tensors (face crops, SRM residuals, rPPG, audio features, metadata).

Safety:  When more than 50 videos are available and ``--n-videos`` is not
         specified, the runner automatically caps at 5 videos to prevent
         accidental multi-hour runs.

Usage (CLI):
    python pipeline_runner.py \\
        --datasets-root  datasets/ \\
        --output-dir     output/   \\
        --n-videos       100

Usage (Python):
    from pipeline_runner import PipelineRunner

    runner = PipelineRunner(
        datasets_root="datasets/",
        output_dir="output/",
    )
    results = runner.run(n_videos=100)
    # results -> {"success": 95, "failed": 5, "skipped": 0}
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional

from video_preprocessor import VideoPreprocessor
from face_tracker import FaceTracker
from feature_extractor import FeatureExtractor
from audio_processor import AudioProcessor
from dataset_manager import DatasetManager


# ======================================================================
#  Logging setup
# ======================================================================
log_dir = Path("pipeline_run_logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "pipeline_run.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ======================================================================
#  Pipeline Runner
# ======================================================================
class PipelineRunner:
    """
    End-to-end extraction pipeline.

    Parameters
    ----------
    datasets_root : str
        Root directory that contains the dataset folders.
    output_dir : str
        Where ``.pt`` output files are saved (auto-created).
    device : str or None
        "cuda" / "cpu".  Auto-detected when None.
    target_fps : int
        Frame rate for extraction (passed to VideoPreprocessor).
    target_res : tuple (width, height)
        Spatial resolution (passed to VideoPreprocessor).
    face_crop_size : tuple (width, height)
        Face crop size (passed to FeatureExtractor).
    """

    def __init__(self, datasets_root: str, output_dir: str,
                 device: str = None, target_fps: int = 25,
                 target_res: tuple = (1280, 720),
                 face_crop_size: tuple = (224, 224)):

        self.datasets_root = datasets_root
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- pipeline components ---
        self.preprocessor    = VideoPreprocessor(target_fps=target_fps, target_res=target_res)
        self.tracker         = FaceTracker(device=device)
        self.extractor       = FeatureExtractor(target_size=face_crop_size)
        self.audio_processor = AudioProcessor()
        self.dataset_manager = DatasetManager(datasets_root)

        datasets_found = [h.name for h in self.dataset_manager.handlers]
        logger.info(f"Pipeline ready  |  output -> {output_dir}")
        logger.info(f"Datasets found: {datasets_found}")

    # ------------------------------------------------------------------
    #  Process a single video
    # ------------------------------------------------------------------
    def process_video(self, entry: Dict) -> Optional[str]:
        """
        Run one video through every pipeline stage and save a ``.pt`` file.

        Returns the output path on success, or ``None`` on failure.
        """
        video_path   = entry["path"]
        video_name   = Path(video_path).stem
        dataset_name = entry["dataset"]

        out_dir  = self.output_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_name}.pt"

        # skip already-processed videos
        if out_path.exists():
            logger.info(f"  skip (exists)  {video_name}")
            return str(out_path)

        try:
            t0 = time.time()

            # 1. frame extraction
            frames, _indices, original_fps = self.preprocessor.extract_frames(video_path)
            if len(frames) == 0:
                logger.warning(f"  WARN  no frames  {video_path}")
                return None

            # 2. face tracking
            tracking = self.tracker.process_frames(frames)
            bboxes   = tracking["primary_subject_boxes"]

            # 3. feature extraction
            features = self.extractor.process_sequence(frames, bboxes)

            # 4. audio features
            audio = self.audio_processor.extract_features(video_path)

            # 5. assemble & save .pt tensor dict
            output = {
                "video_name":        video_name,
                "dataset":           dataset_name,
                "label":             entry["label"],
                "manipulation_type": entry.get("manipulation_type"),
                "face_crops":        torch.from_numpy(features["face_crops"]).float(),
                "residual_maps":     torch.from_numpy(features["residual_maps"]).float(),
                "rppg_signals":      torch.from_numpy(features["rppg_signals"]).float(),
                "landmarks":         torch.from_numpy(features["landmarks"]).float(),
                "pose_angles":       torch.from_numpy(features["pose_angles"]).float(),
                "bboxes":            torch.from_numpy(bboxes).float(),
                "mfcc":              torch.from_numpy(audio["mfcc"]).float(),
                "mel_spectrogram":   torch.from_numpy(audio["mel_spectrogram"]).float(),
                "has_audio":         audio["has_audio"],
                "num_frames":        len(frames),
                "original_fps":      original_fps,
            }
            torch.save(output, str(out_path))

            dt = time.time() - t0
            logger.info(
                f"  OK  {video_name:30s}  {dataset_name:18s}  "
                f"{entry['label']:4s}  {len(frames):4d} fr  {dt:5.1f}s"
            )
            return str(out_path)

        except Exception as exc:
            logger.error(f"  FAIL  {video_path}  --  {exc}")
            return None

    # ------------------------------------------------------------------
    #  Batch run
    # ------------------------------------------------------------------
    def run(self, n_videos: Optional[int] = None, balanced: bool = True,
            dataset_filter: Optional[str] = None):
        """
        Process a batch of videos.

        Parameters
        ----------
        n_videos : int or None
            How many videos to process.  None = all (capped at 5 for safety
            when the catalogue exceeds 50 videos).
        balanced : bool
            Distribute the quota evenly across (dataset, label) groups.
        dataset_filter : str or None
            Only process videos from this dataset.
        """
        summary = self.dataset_manager.summary()

        # --- pretty-print dataset overview ---
        logger.info("")
        logger.info("=" * 62)
        logger.info("  PIPELINE START")
        logger.info("-" * 62)
        logger.info(f"  Total videos available : {summary['total_videos']}")
        for name, info in summary["datasets"].items():
            logger.info(
                f"    {name:20s}  REAL={info['REAL']:>5d}  "
                f"FAKE={info['FAKE']:>5d}  audio={'yes' if info['has_audio'] else 'no'}"
            )
        logger.info("=" * 62)

        # --- select entries ---
        # 1. Filter
        if dataset_filter:
            entries = self.dataset_manager.filter_by_dataset(dataset_filter)
        else:
            entries = self.dataset_manager.get_all_entries()

        # 2. Sample / Cap
        total = len(entries)
        if n_videos:
            if dataset_filter:
                import random
                rng = random.Random(42)
                rng.shuffle(entries)
                entries = entries[:n_videos]
            else:
                entries = self.dataset_manager.sample_subset(n_videos, balanced=balanced)
        else:
            if total > 50:
                logger.warning(
                    f"  No --n-videos specified with {total} videos.  "
                    f"Safety cap: processing 5.  Use --n-videos to override."
                )
                if dataset_filter:
                    import random
                    rng = random.Random(42)
                    rng.shuffle(entries)
                    entries = entries[:5]
                else:
                    entries = self.dataset_manager.sample_subset(5, balanced=balanced)

        logger.info(f"  Processing {len(entries)} video(s) ...\n")

        # --- process loop ---
        results = {"success": 0, "failed": 0, "skipped": 0}
        for i, entry in enumerate(entries, 1):
            p = Path(entry['path'])
            # e.g., Celeb-DF-v2 / Celeb-real / id50_0001.mp4
            display_path = Path(entry['dataset']) / p.parent.name / p.name
            logger.info(f"[{i}/{len(entries)}]  {display_path}")
            out = self.process_video(entry)
            if out:
                results["success"] += 1
            else:
                results["failed"] += 1

        # --- summary ---
        logger.info("")
        logger.info("=" * 62)
        logger.info(
            f"  DONE  |  success={results['success']}  "
            f"failed={results['failed']}  skipped={results['skipped']}"
        )
        logger.info("=" * 62)
        return results


# ======================================================================
#  CLI entry point
# ======================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1 Deepfake Detection – Extraction Pipeline",
    )
    parser.add_argument("--datasets-root", required=True,
                        help="Root directory containing dataset folders")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write .pt output files")
    parser.add_argument("--n-videos", type=int, default=None,
                        help="Number of videos to process (default: safety-capped)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Only process videos from this dataset name")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: 'cuda' or 'cpu' (auto-detected)")

    args = parser.parse_args()

    runner = PipelineRunner(
        datasets_root=args.datasets_root,
        output_dir=args.output_dir,
        device=args.device,
    )
    results = runner.run(n_videos=args.n_videos, dataset_filter=args.dataset)
    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
