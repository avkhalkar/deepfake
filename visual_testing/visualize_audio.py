"""
visualize_audio.py
------------------
Generates a visual validation plot for audio features extracted from a Phase 1 ``.pt`` file.

The script loads a single ``.pt`` tensor payload and produces a 2-row matplotlib figure:
    Row 1: MFCC features (13 × frames).
    Row 2: Mel spectrogram (128 × frames).

If the dataset does not support audio (e.g., non-DFDC), the plot visually indicates
that audio features are missing/unsupported.

Output images are saved to:  ``visual_test_logs/<video_name>_audio_validation.png``

Usage (CLI):
    # From the project root (deepfake/):
    python visual_testing/visualize_audio.py  test_output/DFDC/video_name.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def visualize_audio_file(pt_path: str):
    """
    Load a ``.pt`` tensor payload and render an audio validation figure.
    """
    pt_path = Path(pt_path)

    if not pt_path.exists():
        print(f"  [ERROR]  File not found: {pt_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"  AUDIO VALIDATION")
    print(f"{'=' * 60}")
    print(f"  Loading : {pt_path}")

    data = torch.load(str(pt_path), weights_only=False)

    video_name = data.get("video_name", "?")
    dataset = data.get("dataset", "?")
    label = data.get("label", "?")
    has_audio = data.get("has_audio", False)
    manip_type = data.get('manipulation_type', '')

    fig = plt.figure(figsize=(14, 8))
    
    title_str = f"Audio Validation  |  {dataset}  |  {video_name}  |  {label}"
    if manip_type:
        title_str += f" ({manip_type})"
        
    fig.suptitle(title_str, fontsize=14, fontweight="bold")

    if not has_audio:
        # Display clear message that audio is not supported
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"AUDIO NOT SUPPORTED\nDataset: {dataset}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=24, color='red', weight='bold', transform=ax.transAxes)
        ax.axis('off')
        print(f"  [INFO] Dataset {dataset} does not support audio.")
    else:
        mfcc = data.get("mfcc")
        mel_spectrogram = data.get("mel_spectrogram")

        if mfcc is not None:
            mfcc = mfcc.cpu().numpy()
        if mel_spectrogram is not None:
            mel_spectrogram = mel_spectrogram.cpu().numpy()

        if mfcc is None or mel_spectrogram is None:
            print("  [ERROR]  Missing audio tensors (mfcc / mel_spectrogram).")
            plt.close(fig)
            return

        print(f"  MFCC    : {mfcc.shape}")
        print(f"  Mel Spec: {mel_spectrogram.shape}")

        # Row 1: MFCC
        ax1 = plt.subplot(2, 1, 1)
        im1 = ax1.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title("MFCC Features (13 Coefficients)", fontsize=11)
        ax1.set_ylabel("Coefficients")
        fig.colorbar(im1, ax=ax1)

        # Row 2: Mel Spectrogram
        ax2 = plt.subplot(2, 1, 2)
        # The tensor is already in Log-Mel (dB) scale, ranging roughly from -80 to 0.
        # Plot directly without additional log conversions.
        im2 = ax2.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='magma')
        ax2.set_title("Log Mel Spectrogram (128 Bands)", fontsize=11)
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Mel Bands")
        fig.colorbar(im2, ax=ax2)

    plt.tight_layout()

    out_dir = Path("visual_test_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = out_dir / f"{pt_path.stem}_audio_validation.png"

    plt.savefig(str(out_filename), dpi=150)
    plt.close(fig)

    print(f"  [OK]  Plot saved → {out_filename}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a visual validation plot for audio features from a Phase 1 .pt file.",
        epilog="Example:  python visual_testing/visualize_audio.py test_output/DFDC/video.pt",
    )
    parser.add_argument(
        "pt_file",
        type=str,
        help="Path to the .pt file (e.g., test_output/DFDC/video_name.pt)",
    )
    args = parser.parse_args()

    visualize_audio_file(args.pt_file)
