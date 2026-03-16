"""
run_stage3.py
-------------
Stage 3 stress-test harness.

Builds a **uniform, balanced** sample across all four datasets and runs the
pipeline in configurable chunks with per-chunk progress summaries.

Distribution (default 50 videos):
    Celeb-DF-v2     12  (6 Real / 6 Fake)
    DFDC            12  (6 Real / 6 Fake)        ← audio validation
    DeeperForensics 12  (6 Real / 6 Fake)
    FaceForensics++ 14  (4 Real / 10 Fake)       ← 2 per manipulation type

Usage:
    # Dry-run with 5 videos (1 per dataset + 1 extra FF++)
    python scripts/run_stage3.py --datasets-root datasets/ --output-dir test_output/ --dry-run

    # Full 50-video stress test
    python scripts/run_stage3.py --datasets-root datasets/ --output-dir test_output/

    # Custom count with chunking
    python scripts/run_stage3.py --datasets-root datasets/ --output-dir test_output/ --n-videos 50 --chunk-size 10
"""

import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict

# Add src/ to path so we can import pipeline components
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_manager import DatasetManager
from pipeline_runner import PipelineRunner


# ======================================================================
#  Logging
# ======================================================================
log_dir = PROJECT_ROOT / "pipeline_run_logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "stage3_run.log", mode="a"),
    ],
)
logger = logging.getLogger("stage3")


# ======================================================================
#  Uniform sampling logic
# ======================================================================
def build_uniform_sample(dm: DatasetManager, n_videos: int, seed: int = 42) -> List[Dict]:
    """
    Build a sample with explicit distribution guarantees.

    For 50 videos (default):
        Celeb-DF-v2      → 6 Real, 6 Fake             = 12
        DFDC             → 6 Real, 6 Fake             = 12
        DeeperForensics  → 6 Real, 6 Fake             = 12
        FaceForensics++  → 4 Real, 2 per manip type   = 14
                                                  Total = 50

    For dry-run (5 videos):
        1 per dataset (round-robin Real/Fake) + 1 extra = 5
    """
    rng = random.Random(seed)
    all_entries = dm.get_all_entries()

    # Group by dataset
    by_dataset: Dict[str, List[Dict]] = {}
    for e in all_entries:
        by_dataset.setdefault(e["dataset"], []).append(e)

    if n_videos <= 5:
        return _build_dry_run_sample(by_dataset, n_videos, rng)

    return _build_full_sample(by_dataset, n_videos, rng)


def _build_dry_run_sample(by_dataset: dict, n: int, rng: random.Random) -> List[Dict]:
    """Pick 1 video per dataset, alternating Real/Fake, up to n."""
    sample = []
    use_real = True
    for ds_name in sorted(by_dataset.keys()):
        entries = by_dataset[ds_name]
        target_label = "REAL" if use_real else "FAKE"
        candidates = [e for e in entries if e["label"] == target_label]
        if not candidates:
            candidates = entries  # fallback
        rng.shuffle(candidates)
        sample.append(candidates[0])
        use_real = not use_real
        if len(sample) >= n:
            break

    # If we still need more, grab from the largest dataset
    if len(sample) < n:
        remaining = n - len(sample)
        picked_paths = {e["path"] for e in sample}
        pool = [e for e in sum(by_dataset.values(), []) if e["path"] not in picked_paths]
        rng.shuffle(pool)
        sample.extend(pool[:remaining])

    return sample[:n]


def _build_full_sample(by_dataset: dict, n: int, rng: random.Random) -> List[Dict]:
    """
    Build a uniform distribution for the full run.

    Allocation for 50 videos:
        4 datasets → 12 each = 48, remaining 2 go to FF++ (most categories)
        Within each dataset: 50/50 Real/Fake split
        FF++ exception: 4 Real + 2 from each of 5 manipulation types = 14
    """
    num_datasets = len(by_dataset)
    if num_datasets == 0:
        return []

    # Base allocation: equal split across datasets
    base_per_ds = n // num_datasets
    remainder = n % num_datasets

    # Give remainder to FaceForensics++ (most diverse)
    allocations = {}
    for ds_name in sorted(by_dataset.keys()):
        allocations[ds_name] = base_per_ds
    if "FaceForensics++" in allocations:
        allocations["FaceForensics++"] += remainder
    else:
        # Spread remainder across datasets
        for i, ds_name in enumerate(sorted(by_dataset.keys())):
            if i < remainder:
                allocations[ds_name] += 1

    sample = []

    for ds_name, target_count in allocations.items():
        entries = by_dataset.get(ds_name, [])
        if not entries:
            continue

        if ds_name == "FaceForensics++":
            sample.extend(_sample_ff_plus_plus(entries, target_count, rng))
        else:
            sample.extend(_sample_balanced_real_fake(entries, target_count, rng))

    rng.shuffle(sample)
    return sample[:n]


def _sample_balanced_real_fake(entries: List[Dict], count: int,
                               rng: random.Random) -> List[Dict]:
    """Pick count videos with a 50/50 Real/Fake split."""
    reals = [e for e in entries if e["label"] == "REAL"]
    fakes = [e for e in entries if e["label"] == "FAKE"]
    rng.shuffle(reals)
    rng.shuffle(fakes)

    n_real = count // 2
    n_fake = count - n_real

    picked = reals[:n_real] + fakes[:n_fake]

    # Backfill if one label is short
    if len(picked) < count:
        picked_paths = {e["path"] for e in picked}
        all_remaining = [e for e in entries if e["path"] not in picked_paths]
        rng.shuffle(all_remaining)
        picked.extend(all_remaining[:count - len(picked)])

    return picked[:count]


def _sample_ff_plus_plus(entries: List[Dict], count: int,
                         rng: random.Random) -> List[Dict]:
    """
    Sample FF++ with category awareness.

    Strategy for 14 videos:
        4 Real (from original_sequences)
        2 each from: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures = 10
    """
    reals = [e for e in entries if e["label"] == "REAL"]
    rng.shuffle(reals)

    # Group fakes by manipulation_type
    fakes_by_type: Dict[str, List[Dict]] = {}
    for e in entries:
        if e["label"] == "FAKE":
            mt = e.get("manipulation_type") or "Unknown"
            fakes_by_type.setdefault(mt, []).append(e)

    manip_types = sorted(fakes_by_type.keys())
    n_manip_types = len(manip_types) if manip_types else 1

    # Allocate: fakes get 2 per type, rest goes to real
    n_fake_per_type = 2
    n_total_fake = min(n_fake_per_type * n_manip_types, count)
    n_real = count - n_total_fake

    picked = reals[:n_real]

    for mt in manip_types:
        candidates = fakes_by_type[mt]
        rng.shuffle(candidates)
        picked.extend(candidates[:n_fake_per_type])

    # Backfill if short
    if len(picked) < count:
        picked_paths = {e["path"] for e in picked}
        all_remaining = [e for e in entries if e["path"] not in picked_paths]
        rng.shuffle(all_remaining)
        picked.extend(all_remaining[:count - len(picked)])

    return picked[:count]


# ======================================================================
#  Distribution report
# ======================================================================
def print_distribution(sample: List[Dict]):
    """Print a human-readable breakdown of the sample distribution."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STAGE 3 SAMPLE DISTRIBUTION")
    logger.info("-" * 70)

    by_ds: Dict[str, Dict] = {}
    for e in sample:
        ds = e["dataset"]
        if ds not in by_ds:
            by_ds[ds] = {"REAL": 0, "FAKE": 0, "UNKNOWN": 0,
                         "has_audio": e.get("has_audio", False),
                         "manip_types": set()}
        by_ds[ds][e["label"]] += 1
        mt = e.get("manipulation_type")
        if mt:
            by_ds[ds]["manip_types"].add(mt)

    for ds_name in sorted(by_ds.keys()):
        info = by_ds[ds_name]
        total = info["REAL"] + info["FAKE"] + info["UNKNOWN"]
        audio_str = "🔊 audio" if info["has_audio"] else "🔇 silent"
        logger.info(
            f"  {ds_name:20s}  {total:3d} videos  "
            f"(R={info['REAL']:2d}  F={info['FAKE']:2d})  {audio_str}"
        )
        if info["manip_types"]:
            for mt in sorted(info["manip_types"]):
                mt_count = sum(1 for e in sample
                               if e["dataset"] == ds_name
                               and e.get("manipulation_type") == mt)
                logger.info(f"    |- {mt:20s}  {mt_count} videos")

    logger.info(f"\n  TOTAL: {len(sample)} videos")
    logger.info("=" * 70)


# ======================================================================
#  Chunked execution
# ======================================================================
def run_chunked(runner: PipelineRunner, sample: List[Dict],
                chunk_size: int, pause: float):
    """
    Process videos in chunks with progress summaries between each chunk.
    """
    total = len(sample)
    n_chunks = (total + chunk_size - 1) // chunk_size
    overall = {"success": 0, "failed": 0, "skipped": 0}
    global_start = time.time()

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        chunk = sample[start:end]

        logger.info("")
        logger.info(f"  -- CHUNK {chunk_idx + 1}/{n_chunks}  "
                     f"(videos {start + 1}–{end} of {total}) --")

        chunk_start = time.time()
        for i, entry in enumerate(chunk, 1):
            video_name = Path(entry["path"]).stem
            logger.info(f"  [{start + i}/{total}]  {entry['dataset']} / {video_name}")
            out = runner.process_video(entry)
            if out:
                overall["success"] += 1
            else:
                overall["failed"] += 1

        chunk_dt = time.time() - chunk_start
        elapsed_total = time.time() - global_start
        rate = overall["success"] / (elapsed_total / 3600) if elapsed_total > 0 else 0

        logger.info(f"  -- chunk done in {chunk_dt:.1f}s  |  "
                     f"cumulative: {overall['success']} OK, {overall['failed']} FAIL  |  "
                     f"rate: {rate:.0f} videos/hr --")

        # Pause between chunks (except after last)
        if chunk_idx < n_chunks - 1 and pause > 0:
            logger.info(f"  (pausing {pause:.0f}s before next chunk...)")
            time.sleep(pause)

    total_dt = time.time() - global_start
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  STAGE 3 COMPLETE  |  {total_dt:.1f}s total  "
                f"|  success={overall['success']}  failed={overall['failed']}")
    logger.info("=" * 70)

    return overall


# ======================================================================
#  Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 Stress Test – Uniform Distribution Runner",
    )
    parser.add_argument("--datasets-root", type=str, default="datasets",
                        help="Root directory containing dataset folders")
    parser.add_argument("--output-dir", type=str, default="test_output",
                        help="Where to write .pt output files")
    parser.add_argument("--n-videos", type=int, default=50,
                        help="Total number of videos to process (default: 50)")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Videos per processing chunk (default: 10)")
    parser.add_argument("--pause", type=float, default=3.0,
                        help="Seconds to pause between chunks (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: 'cuda' or 'cpu'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on only 5 videos for quick validation")

    args = parser.parse_args()

    # Resolve paths relative to project root
    datasets_root = str(PROJECT_ROOT / args.datasets_root)
    output_dir = str(PROJECT_ROOT / args.output_dir)

    n_videos = 5 if args.dry_run else args.n_videos

    logger.info(f"Stage 3 harness starting  |  target={n_videos} videos")

    # 1. Build uniform sample
    dm = DatasetManager(datasets_root)
    sample = build_uniform_sample(dm, n_videos, seed=args.seed)
    print_distribution(sample)

    # 2. Create pipeline runner
    runner = PipelineRunner(
        datasets_root=datasets_root,
        output_dir=output_dir,
        device=args.device,
    )

    # 3. Run in chunks
    chunk_size = min(args.chunk_size, n_videos)
    results = run_chunked(runner, sample, chunk_size, args.pause)

    print(f"\nResults: {results}")
    return results


if __name__ == "__main__":
    main()
