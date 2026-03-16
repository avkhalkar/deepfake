# 🎭 DeepFake Detection Pipeline

A modular, production-grade preprocessing pipeline for multi-modal deepfake detection. This project extracts **spatial**, **temporal**, **physiological**, and **audio** features from raw video files across four major deepfake datasets, producing standardized PyTorch tensor payloads (`.pt`) ready for downstream model training.

> **Current Status:** Phase 1 (Preprocessing & Feature Extraction) is **complete** and fully tested.  
> Phase 2 (Model Training) is the next milestone.

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Supported Datasets](#-supported-datasets)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Output Schema](#-output-schema)
- [Testing](#-testing)
- [Visual Validation](#-visual-validation)
- [Stress Testing](#-stress-testing)
- [Documentation](#-documentation)
- [Phase 2 Roadmap](#-phase-2-roadmap)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Frame Extraction** | Uniform 25 FPS sampling with 720p resizing via OpenCV |
| **Face Detection & Tracking** | MTCNN detection + Kalman Filter (SORT) with "First-Select, Then-Lock" identity locking |
| **Spatial Features** | 224×224 aligned face crops per frame |
| **High-Frequency Residuals** | SRM (Spatial Rich Model) noise maps to expose manipulation artifacts |
| **Physiological Signals** | rPPG estimation from skin ROI (RGB temporal averages, Savitzky-Golay smoothed) |
| **Audio Features** | 13-coefficient MFCC + 128-band Log-Mel Spectrogram (with graceful zero-tensor fallback for silent datasets) |
| **Multi-Dataset Support** | Unified interface across 4 major datasets with automatic structure discovery |
| **Crash-Safe Resume** | Skips already-processed videos on re-run |
| **Memory Monitoring** | Sidecar script to track RAM usage and detect memory leaks |

---

## 📁 Project Structure

```
deepfake/
├── src/                          # Core pipeline modules
│   ├── video_preprocessor.py     # Frame extraction & resizing
│   ├── face_tracker.py           # MTCNN + Kalman Filter tracking
│   ├── feature_extractor.py      # Spatial crops, SRM residuals, rPPG
│   ├── audio_processor.py        # MFCC & Mel-spectrogram extraction
│   ├── dataset_manager.py        # Unified dataset interface (4 datasets)
│   └── pipeline_runner.py        # End-to-end orchestration & CLI
│
├── scripts/                      # Operational scripts
│   ├── run_stage3.py             # Stress test harness (uniform sampling)
│   ├── stage3_monitor.py         # RAM monitoring sidecar
│   └── stage3_health_report.py   # Post-run validation & summary
│
├── visual_testing/               # Visual validation tools
│   ├── visualize_pt.py           # Plot face crops, SRM residuals, rPPG
│   └── visualize_audio.py        # Plot MFCC & Mel spectrogram features
│
├── tests/                        # Automated test suite
│   ├── unit/                     # Component-level tests
│   │   ├── test_video_preprocessor.py
│   │   ├── test_face_tracker.py
│   │   ├── test_feature_extractor.py
│   │   ├── test_dataset_manager.py
│   │   └── test_audio_processor.py
│   └── integration/              # End-to-end tests
│       ├── test_pipeline_integration.py
│       └── test_output_files.py
│
├── datasets/                     # Dataset root (not tracked in git)
│   ├── Celeb-DF-v2/
│   ├── dfdc/
│   ├── deeperforensics/
│   └── face_forensics/
│
├── models/                       # Pre-trained model weights
├── docs/                         # Project documentation
│   ├── checklist.md              # Phase 1 completion checklist
│   ├── phase1_code_architecture.md
│   └── project_structure.md
│
├── test_output/                  # Extracted .pt files (not tracked)
├── test_logs/                    # Test run logs (not tracked)
├── visual_test_logs/             # Visual validation plots (not tracked)
├── pipeline_run_logs/            # Extraction logs (not tracked)
│
├── USAGE.md                      # Detailed usage guide
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 🗂 Supported Datasets

| Dataset | Videos | Audio | Labels | Manipulation Types |
|---|---|---|---|---|
| **DFDC** | ~120K | ✅ Yes | Real / Fake | — |
| **Celeb-DF-v2** | ~6K | ❌ No | Real / Fake | — |
| **FaceForensics++** | ~5K | ❌ No | Real / Fake | Deepfakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter, Original |
| **DeeperForensics** | ~60K | ❌ No | Real / Fake | — |

The `DatasetManager` automatically discovers and parses each dataset's unique directory structure and metadata format. It provides:
- Unified video entry dictionaries with standardized fields
- Balanced sampling across datasets and Real/Fake classes
- Automatic `has_audio` and `manipulation_type` tagging

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (recommended, but CPU mode is supported)
- **FFmpeg** (must be on system PATH for audio extraction)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd deepfake

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Place your datasets under the `datasets/` directory following this structure:

```
datasets/
├── Celeb-DF-v2/                                    # Celeb-DF-v2
│   ├── Celeb-real/                                # Real Videos
│   ├── Celeb-synthesis/                           # Deepfake Videos
│   ├── YouTube-real/                              # Real Videos
│   └── List_of_testing_videos.txt
├── dfdc/                                            # Facebook DFDC
│   └── train_sample_videos/
│       ├── metadata.json
│       └── *.mp4
├── deeperforensics/                                 # DeeperForensics-1.0
│   ├── source_videos_part_03/                     # Source: Real Videos
│   ├── manipulated_videos_part_01/                # Manipulated: Deepfake Videos
│   └── lists/
│       ├── manipulated_videos_distortions_meta/     # Distortion metadata .txt files
│       ├── manipulated_videos_lists/                # Video list manifests .txt files
│       ├── source_videos_lists/                     # Source video lists .txt files
│       └── splits/                                  # train.txt, val.txt, test.txt
└── face_forensics/                                  # FaceForensics++
    ├── original-sequences-c23-videos/               # Real Videos
    ├── manipulated-sequences-Deepfakes-c23-videos/    # Deepfake Videos
    ├── manipulated-sequences-Face2Face-c23-videos/    # Face2Face Videos
    ├── manipulated-sequences-FaceShifter-c23-videos/  # FaceShifter Videos
    ├── manipulated-sequences-FaceSwap-c23-videos/     # FaceSwap Videos
    ├── manipulated-sequences-NeuralTextures-c23-videos/ # NeuralTextures Videos
    └── csv/                                           # Metadata files
```

---

## 💻 Usage

### Run the Full Pipeline

```bash
# Process 5 balanced videos (default safety cap)
python src/pipeline_runner.py --datasets-root datasets/ --output-dir test_output/ --n-videos 5

# Process a specific number of videos
python src/pipeline_runner.py --datasets-root datasets/ --output-dir test_output/ --n-videos 50

# Target a single dataset
python src/pipeline_runner.py --datasets-root datasets/ --output-dir test_output/ --dataset DFDC
```

### Load an Extracted Tensor

```python
import torch

data = torch.load("test_output/DFDC/afoovlsmtx.pt", weights_only=False)

print(data.keys())                  # All 16 schema fields
print(data["face_crops"].shape)     # [T, 224, 224, 3]
print(data["label"])                # 'REAL' or 'FAKE'
print(data["has_audio"])            # True (DFDC has audio)
print(data["mfcc"].shape)           # [13, T_audio]
print(data["mel_spectrogram"].shape)# [128, T_audio]
```

> For detailed component-level usage (VideoPreprocessor, FaceTracker, FeatureExtractor, AudioProcessor, DatasetManager), see **[USAGE.md](USAGE.md)**.

---

## 📦 Output Schema

Each `.pt` file is a Python dictionary with the following **16 fields**:

| # | Key | Shape / Type | Description |
|---|---|---|---|
| 1 | `video_name` | `str` | Source video filename (without extension) |
| 2 | `dataset` | `str` | Dataset name (e.g., `"DFDC"`, `"FaceForensics++"`) |
| 3 | `label` | `str` | `"REAL"` or `"FAKE"` |
| 4 | `manipulation_type` | `str` | Sub-category (e.g., `"Deepfakes"`) or empty for non-FF++ |
| 5 | `num_frames` | `int` | Number of extracted frames |
| 6 | `source_fps` | `float` | Original video FPS |
| 7 | `face_crops` | `Tensor [T, 224, 224, 3]` | Aligned RGB face crops |
| 8 | `residual_maps` | `Tensor [T, 224, 224, 1]` | SRM high-frequency noise maps |
| 9 | `rppg_signals` | `Tensor [T, 3]` | Smoothed rPPG (R, G, B channels) |
| 10 | `bboxes` | `Tensor [T, 4]` | Face bounding boxes (x1, y1, x2, y2) |
| 11 | `landmarks` | `Tensor [T, 68, 2]` | 68-point facial landmarks |
| 12 | `pose_angles` | `Tensor [T, 3]` | Head pose (pitch, yaw, roll) |
| 13 | `has_audio` | `bool` | Whether audio features are valid |
| 14 | `mfcc` | `Tensor [13, T_audio]` | 13 MFCC coefficients |
| 15 | `mel_spectrogram` | `Tensor [128, T_audio]` | 128-band Log-Mel spectrogram (dB scale) |
| 16 | `video_path` | `str` | Absolute path to the source video |

> **Note:** For non-audio datasets (`has_audio=False`), `mfcc` and `mel_spectrogram` are zero tensors of shape `[13, 1]` and `[128, 1]` respectively.

---

## 🧪 Testing

### Run the Full Test Suite

```bash
pytest tests/ -v
```

### Unit Tests (`tests/unit/`)

| Test File | What It Validates |
|---|---|
| `test_video_preprocessor.py` | Frame extraction rates, resolution resizing |
| `test_face_tracker.py` | SORT tracking, identity locking, IOU matching |
| `test_feature_extractor.py` | Crop dimensions, SRM residuals, rPPG shapes |
| `test_dataset_manager.py` | Dataset discovery, balanced sampling |
| `test_audio_processor.py` | MFCC/Mel extraction, zero-padding fallback |

### Integration Tests (`tests/integration/`)

| Test File | What It Validates |
|---|---|
| `test_pipeline_integration.py` | Full end-to-end pipeline on real dataset videos |
| `test_output_files.py` | Schema validation on all generated `.pt` files |

---

## 🎨 Visual Validation

Visual testing scripts generate matplotlib plots to visually verify extraction quality.

```bash
# Spatial validation (face crops, SRM residuals, rPPG signals)
python visual_testing/visualize_pt.py test_output/DFDC/video_name.pt

# Audio validation (MFCC & Mel spectrogram, or "AUDIO NOT SUPPORTED" warning)
python visual_testing/visualize_audio.py test_output/DFDC/video_name.pt
```

Output plots are saved to `visual_test_logs/`.

- **FaceForensics++** plots automatically display the manipulation sub-type (e.g., `Deepfakes`, `FaceSwap`) in the figure title.
- **Non-audio datasets** render a clear red warning: *"AUDIO NOT SUPPORTED"*.

---

## 🔥 Stress Testing

The pipeline has been validated with a **50-video stress test** covering all 4 datasets with balanced Real/Fake distribution.

```bash
# Start the RAM monitor (background process)
python scripts/stage3_monitor.py &

# Run the stress test
python scripts/run_stage3.py --n-videos 50 --chunk-size 10

# Generate a health report
python scripts/stage3_health_report.py
```

### Verified Results

| Metric | Value |
|---|---|
| Videos Processed | 50 |
| Success Rate | 100% (0 failures) |
| Processing Time | ~20 minutes |
| Avg. Speed | ~24 seconds/video |
| Peak RAM | 5.8 GB |
| Memory Leak | ❌ None detected |
| Schema Integrity | ✅ 100% pass |
| Audio Flag Accuracy | ✅ 100% correct |

---

## 📚 Documentation

| Document | Description |
|---|---|
| [USAGE.md](USAGE.md) | Detailed usage guide for all pipeline components |
| [docs/checklist.md](docs/checklist.md) | Phase 1 completion checklist (all items ✅) |
| [docs/phase1_code_architecture.md](docs/phase1_code_architecture.md) | Architecture design and module breakdown |
| [docs/project_structure.md](docs/project_structure.md) | Directory structure specification |

---

## 🗺 Phase 2 Roadmap

Phase 1 delivers clean, standardized tensor payloads. The next phase focuses on **model training and evaluation**:

1. **PyTorch Dataset / DataLoader** — Build a `torch.utils.data.Dataset` class that loads `.pt` files and serves batches for training.
2. **Multi-Modal Fusion Model** — Design a model architecture that fuses spatial (face crops + SRM), temporal (rPPG), and audio (MFCC + Mel) streams.
3. **Training Loop** — Implement training with cross-entropy loss, learning rate scheduling, and mixed-precision training.
4. **Evaluation Pipeline** — Compute AUC, accuracy, and per-dataset performance metrics.
5. **Ablation Studies** — Measure the contribution of each modality stream.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
