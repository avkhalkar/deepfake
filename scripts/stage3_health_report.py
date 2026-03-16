"""
stage3_health_report.py
-----------------------
Post-run validation and summary for Stage 3 stress test.

Reads all .pt files from the output directory, validates their schema,
reads the memory CSV (if available), and prints a comprehensive pass/fail
health report.

Usage:
    python scripts/stage3_health_report.py
    python scripts/stage3_health_report.py --output-dir test_output/ --log-dir pipeline_run_logs/
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
#  Schema validation (mirrors test_output_files.py logic)
# ======================================================================
EXPECTED_KEYS = {
    "video_name", "dataset", "label", "has_audio", "num_frames",
    "face_crops", "residual_maps", "rppg_signals", "pose_angles", "bboxes",
    "mfcc", "mel_spectrogram",
}


def validate_pt_file(pt_path: Path) -> Tuple[bool, str]:
    """
    Validate a single .pt file against the Phase 1 schema.

    Returns (passed: bool, message: str).
    """
    try:
        data = torch.load(str(pt_path), weights_only=False)
    except Exception as e:
        return False, f"Failed to load: {e}"

    # Key check
    missing = EXPECTED_KEYS - set(data.keys())
    if missing:
        return False, f"Missing keys: {missing}"

    # Metadata types
    if not isinstance(data.get("video_name"), str):
        return False, "video_name is not a string"
    if not isinstance(data.get("dataset"), str):
        return False, "dataset is not a string"
    if data.get("label") not in ("REAL", "FAKE", "UNKNOWN"):
        return False, f"Invalid label: {data.get('label')}"
    if not isinstance(data.get("has_audio"), bool):
        return False, "has_audio is not a bool"

    T = data.get("num_frames")
    if not isinstance(T, int) or T <= 0:
        return False, f"Invalid num_frames: {T}"

    # Tensor shapes
    shape_checks = [
        ("face_crops", (T, 224, 224, 3)),
        ("rppg_signals", (T, 3)),
        ("pose_angles", (T, 3)),
        ("bboxes", (T, 4)),
    ]
    for key, expected_shape in shape_checks:
        actual = data[key].shape
        if actual != expected_shape:
            return False, f"{key} shape {actual} != expected {expected_shape}"

    # Residual maps: [T, 224, 224, 1] OR [T, 224, 224, 3] (both are valid)
    rm_shape = data["residual_maps"].shape
    if rm_shape not in ((T, 224, 224, 1), (T, 224, 224, 3)):
        return False, f"residual_maps shape {rm_shape} is invalid"

    # Audio validation
    if data["has_audio"]:
        if data["mfcc"].numel() == 0:
            return False, "has_audio=True but mfcc is empty"
        if data["mel_spectrogram"].numel() == 0:
            return False, "has_audio=True but mel_spectrogram is empty"
        if data["mfcc"].shape[0] != 13:
            return False, f"MFCC has {data['mfcc'].shape[0]} coefficients, expected 13"
        if data["mel_spectrogram"].shape[0] != 128:
            return False, f"Mel spec has {data['mel_spectrogram'].shape[0]} bands, expected 128"
    else:
        if data["mfcc"].numel() != 0:
            return False, f"has_audio=False but mfcc has shape {data['mfcc'].shape}"
        if data["mel_spectrogram"].numel() != 0:
            return False, f"has_audio=False but mel_spectrogram has shape {data['mel_spectrogram'].shape}"

    # Dtype check
    for key in ["face_crops", "residual_maps", "rppg_signals", "pose_angles",
                 "bboxes", "mfcc", "mel_spectrogram"]:
        if data[key].numel() > 0 and data[key].dtype != torch.float32:
            return False, f"{key} dtype is {data[key].dtype}, expected float32"

    return True, "OK"


# ======================================================================
#  Memory analysis
# ======================================================================
def analyze_memory_csv(csv_path: Path) -> Dict:
    """Read the RAM monitor CSV and compute stats."""
    if not csv_path.exists():
        return {"available": False}

    rss_values = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rss = row.get("rss_mb", "0")
                if rss:
                    try:
                        rss_values.append(float(rss))
                    except ValueError:
                        pass
    except Exception:
        return {"available": False}

    if not rss_values:
        return {"available": False}

    # Check for monotonic growth (leak indicator)
    # Look at first 10% vs last 10%
    n = len(rss_values)
    window = max(1, n // 10)
    first_avg = sum(rss_values[:window]) / window
    last_avg = sum(rss_values[-window:]) / window
    growth_mb = last_avg - first_avg

    return {
        "available": True,
        "samples": n,
        "min_mb": min(rss_values),
        "max_mb": max(rss_values),
        "mean_mb": sum(rss_values) / n,
        "first_avg_mb": first_avg,
        "last_avg_mb": last_avg,
        "growth_mb": growth_mb,
        "leak_suspect": growth_mb > 500,  # >500MB growth = suspect
    }


# ======================================================================
#  Report generation
# ======================================================================
def generate_report(output_dir: Path, log_dir: Path) -> str:
    """Generate the full health report and return it as a string."""
    lines = []
    lines.append("=" * 70)
    lines.append("  STAGE 3 HEALTH REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # ── 1. Schema Validation ──
    pt_files = list(output_dir.rglob("*.pt")) if output_dir.exists() else []
    total = len(pt_files)
    passed = 0
    failed = 0
    failures = []

    # Track per-dataset and audio stats
    by_dataset: Dict[str, Dict] = {}

    for pt in sorted(pt_files):
        ok, msg = validate_pt_file(pt)
        if ok:
            passed += 1
            # Load metadata for distribution stats
            try:
                data = torch.load(str(pt), weights_only=False)
                ds = data.get("dataset", "Unknown")
                label = data.get("label", "Unknown")
                has_audio = data.get("has_audio", False)
                manip = data.get("manipulation_type", None)

                if ds not in by_dataset:
                    by_dataset[ds] = {"REAL": 0, "FAKE": 0, "UNKNOWN": 0,
                                       "audio_true": 0, "audio_false": 0,
                                       "manip_types": set()}
                by_dataset[ds][label] = by_dataset[ds].get(label, 0) + 1
                if has_audio:
                    by_dataset[ds]["audio_true"] += 1
                else:
                    by_dataset[ds]["audio_false"] += 1
                if manip:
                    by_dataset[ds]["manip_types"].add(manip)
            except Exception:
                pass
        else:
            failed += 1
            failures.append((pt.name, msg))

    lines.append("── SCHEMA VALIDATION ──")
    lines.append(f"  Total .pt files : {total}")
    lines.append(f"  Passed          : {passed}")
    lines.append(f"  Failed          : {failed}")
    if failures:
        lines.append("  Failures:")
        for name, msg in failures:
            lines.append(f"    ✗ {name}: {msg}")
    lines.append("")

    # ── 2. Distribution Breakdown ──
    lines.append("── DISTRIBUTION BREAKDOWN ──")
    for ds_name in sorted(by_dataset.keys()):
        info = by_dataset[ds_name]
        total_ds = info["REAL"] + info["FAKE"] + info.get("UNKNOWN", 0)
        audio_str = f"audio={info['audio_true']}/{total_ds}"
        lines.append(
            f"  {ds_name:20s}  {total_ds:3d} total  "
            f"(R={info['REAL']:2d}  F={info['FAKE']:2d})  {audio_str}"
        )
        if info["manip_types"]:
            for mt in sorted(info["manip_types"]):
                lines.append(f"    └─ {mt}")
    lines.append("")

    # ── 3. Audio Feature Audit ──
    lines.append("── AUDIO FEATURE AUDIT ──")
    audio_issues = []
    for pt in sorted(pt_files):
        try:
            data = torch.load(str(pt), weights_only=False)
            ds = data.get("dataset", "")
            has_audio = data.get("has_audio", False)

            if ds == "DFDC" and not has_audio:
                audio_issues.append(f"  ⚠ {pt.name}: DFDC video but has_audio=False")
            elif ds != "DFDC" and has_audio:
                audio_issues.append(f"  ⚠ {pt.name}: {ds} video but has_audio=True")
        except Exception:
            pass

    if audio_issues:
        for issue in audio_issues:
            lines.append(issue)
    else:
        lines.append("  ✓ All audio flags consistent (DFDC=True, others=False)")
    lines.append("")

    # ── 4. Memory Analysis ──
    mem_csv = log_dir / "stage3_memory_report.csv"
    mem = analyze_memory_csv(mem_csv)

    lines.append("── MEMORY ANALYSIS ──")
    if mem["available"]:
        lines.append(f"  Samples          : {mem['samples']}")
        lines.append(f"  RSS min          : {mem['min_mb']:.1f} MB")
        lines.append(f"  RSS max          : {mem['max_mb']:.1f} MB")
        lines.append(f"  RSS mean         : {mem['mean_mb']:.1f} MB")
        lines.append(f"  Growth (Δ)       : {mem['growth_mb']:+.1f} MB")
        if mem["leak_suspect"]:
            lines.append("  ⚠ WARNING: RAM grew by >500MB — possible memory leak!")
        else:
            lines.append("  ✓ No memory leak detected")
    else:
        lines.append("  (no memory monitor CSV found — run stage3_monitor.py alongside the pipeline)")
    lines.append("")

    # ── 5. Overall Verdict ──
    lines.append("=" * 70)
    all_pass = (failed == 0) and (not audio_issues) and (
        not mem.get("available") or not mem.get("leak_suspect", False)
    )
    if all_pass:
        lines.append("  ✅ STAGE 3 PASSED")
    else:
        lines.append("  ❌ STAGE 3 HAS ISSUES — review above")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 Post-Run Health Report",
    )
    parser.add_argument("--output-dir", type=str, default="test_output",
                        help="Pipeline output directory containing .pt files")
    parser.add_argument("--log-dir", type=str, default="pipeline_run_logs",
                        help="Directory with memory monitor CSV")

    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    log_dir = PROJECT_ROOT / args.log_dir

    report = generate_report(output_dir, log_dir)
    print(report)

    # Save to file
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "stage3_health_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
