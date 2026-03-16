"""
stage3_monitor.py
-----------------
Sidecar memory monitor for Stage 3 stress test.

Runs in a separate terminal alongside the pipeline and samples system/process
RAM at a fixed interval, writing a CSV time-series to
``pipeline_run_logs/stage3_memory_report.csv``.

Usage:
    # Auto-discover the pipeline_runner process:
    python scripts/stage3_monitor.py

    # Or specify a PID directly:
    python scripts/stage3_monitor.py --pid 12345

    # Custom interval and output directory:
    python scripts/stage3_monitor.py --interval 15 --log-dir pipeline_run_logs/
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required.  Install with:  pip install psutil")
    sys.exit(1)


def find_pipeline_pid() -> int | None:
    """Auto-discover a running pipeline_runner.py process."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            joined = " ".join(cmdline).lower()
            if "pipeline_runner" in joined or "run_stage3" in joined:
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def count_pt_files(output_dir: Path) -> int:
    """Count .pt files already written."""
    if not output_dir.exists():
        return 0
    return len(list(output_dir.rglob("*.pt")))


def monitor(pid: int | None, interval: float, log_dir: Path,
            output_dir: Path, duration: float | None):
    """
    Main monitoring loop.

    Parameters
    ----------
    pid : int or None
        PID to track.  If None, auto-discovers every cycle.
    interval : float
        Seconds between samples.
    log_dir : Path
        Where to write the CSV report.
    output_dir : Path
        Pipeline output directory (to count .pt files for progress).
    duration : float or None
        Max seconds to run.  None = run until Ctrl-C.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "stage3_memory_report.csv"

    # Write header if file doesn't exist yet
    write_header = not csv_path.exists()

    peak_rss_mb = 0.0
    start_time = time.time()

    print(f"[monitor] Logging to {csv_path}")
    print(f"[monitor] Sampling every {interval}s  |  Ctrl-C to stop")
    print()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "elapsed_sec", "pid",
                "rss_mb", "peak_rss_mb",
                "system_ram_percent", "pt_files_written",
            ])

        try:
            while True:
                elapsed = time.time() - start_time

                # Resolve PID each cycle (in case process restarted)
                current_pid = pid or find_pipeline_pid()

                rss_mb = 0.0
                if current_pid:
                    try:
                        proc = psutil.Process(current_pid)
                        mem = proc.memory_info()
                        rss_mb = mem.rss / (1024 * 1024)
                        peak_rss_mb = max(peak_rss_mb, rss_mb)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        current_pid = None

                sys_ram = psutil.virtual_memory().percent
                pt_count = count_pt_files(output_dir)
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                writer.writerow([
                    now_str, f"{elapsed:.1f}", current_pid or "",
                    f"{rss_mb:.1f}", f"{peak_rss_mb:.1f}",
                    f"{sys_ram:.1f}", pt_count,
                ])
                f.flush()

                status = "tracking" if current_pid else "waiting for process"
                print(
                    f"  [{now_str}]  {status:20s}  "
                    f"RSS={rss_mb:7.1f} MB  peak={peak_rss_mb:7.1f} MB  "
                    f"sys={sys_ram:4.1f}%  .pt={pt_count}",
                )

                if duration and elapsed >= duration:
                    print(f"\n[monitor] Duration limit reached ({duration}s).  Stopping.")
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[monitor] Stopped by user.")

    print(f"\n[monitor] Peak RSS: {peak_rss_mb:.1f} MB")
    print(f"[monitor] Report saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 sidecar memory monitor",
    )
    parser.add_argument("--pid", type=int, default=None,
                        help="PID of the pipeline process (auto-detected if omitted)")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Sampling interval in seconds (default: 30)")
    parser.add_argument("--log-dir", type=str, default="pipeline_run_logs",
                        help="Directory for the CSV report")
    parser.add_argument("--output-dir", type=str, default="test_output",
                        help="Pipeline output directory (for .pt progress count)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Max monitoring duration in seconds (default: unlimited)")

    args = parser.parse_args()

    monitor(
        pid=args.pid,
        interval=args.interval,
        log_dir=Path(args.log_dir),
        output_dir=Path(args.output_dir),
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
