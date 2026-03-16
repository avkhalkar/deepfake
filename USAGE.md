# Phase 1 Data Extraction Pipeline Usage Guide

This document explains how to use the Phase 1 deepfake video extraction pipeline. The pipeline takes raw `.mp4` videos from various datasets (Celeb-DF, DFDC, DeeperForensics, FaceForensics++) and converts them into standardized `.pt` (PyTorch) tensor files containing cropped faces, high-pass SRM residuals, physiological rPPG signals, and audio features.

## 1. Quick Start (Running the Entire Pipeline)

The easiest way to process videos is using the `pipeline_runner.py` CLI. Ensure your virtual environment is activated.

```bash
# Process 5 videos (balanced across real/fake and datasets)
python src/pipeline_runner.py --datasets-root datasets/ --output-dir test_output/ --n-videos 5

# Process all videos from a specific dataset (e.g., DFDC)
python src/pipeline_runner.py --datasets-root datasets/ --output-dir test_output/ --dataset DFDC
```

### Safety Cap
If you run without `--n-videos` and you have more than 50 videos in your datasets folder, the runner will **automatically cap at 5 videos** to prevent accidental massive runs that could take weeks. To explicitly process everything, pass `--n-videos 99999` (or the exact total).

### Output Format
The pipeline creates one `.pt` file per video in the `--output-dir`. For a video named `id50_0001.mp4` from Celeb-DF, the output is `test_output/Celeb-DF-v2/id50_0001.pt`. 

You can load and inspect it using PyTorch:

```python
import torch

data = torch.load("test_output/Celeb-DF-v2/id50_0001.pt", weights_only=False)

print(data["label"])              # 'REAL' or 'FAKE'
print(data["face_crops"].shape)   # [T, 224, 224, 3]
print(data["has_audio"])          # False (Celeb-DF has no audio)
```

---

## 2. Using Individual Components

If you need to change how a specific part works or want to test components in isolation, you can import them into your own scripts.

### Video Preprocessor
Extracts frames uniformly from a video at a fixed FPS and resizes them.
```python
from src.video_preprocessor import VideoPreprocessor

preprocessor = VideoPreprocessor(target_fps=25, target_res=(1280, 720))
frames, indices, source_fps = preprocessor.extract_frames("datasets/Celeb-DF-v2/Celeb-real/id50_0001.mp4")
# frames is a numpy array of shape [T, 720, 1280, 3] (RGB)
```

### Face Tracker
Detects multiple faces using MTCNN, tracks them using a Kalman filter, and locks onto the primary subject (largest face) to follow it across frames.
```python
from src.face_tracker import FaceTracker

tracker = FaceTracker(device="cuda") # or "cpu"
tracking_result = tracker.process_frames(frames)
bboxes = tracking_result["primary_subject_boxes"] 
# numpy array of shape [T, 4]. Rows with NaNs mean the face was completely lost.
```

### Feature Extractor
Takes the frames and bounding boxes to extract spatial crops, high-pass SRM residuals, and temporal rPPG signals.
```python
from src.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(target_size=(224, 224))
features = extractor.process_sequence(frames, bboxes)

print(features["face_crops"].shape)    # [T, 224, 224, 3]
print(features["residual_maps"].shape) # [T, 224, 224, 3]
print(features["rppg_signals"].shape)  # [T, 3] (smoothed RGB averages)
```

### Audio Processor
Extracts MFCC and Mel-spectrogram features. Safe to use on silent videos (returns zeroed tensors).
```python
from src.audio_processor import AudioProcessor

audio_proc = AudioProcessor()
audio_features = audio_proc.extract_features("datasets/dfdc/train_sample_videos/eqjscdagiv.mp4")

if audio_features["has_audio"]:
    print(audio_features["mfcc"].shape)            # [13, T]
    print(audio_features["mel_spectrogram"].shape) # [128, T]
```

### Dataset Manager
Auto-discovers and manages your datasets, so you don't have to write custom traversal logic.
```python
from src.dataset_manager import DatasetManager

manager = DatasetManager("datasets/")
print(manager.summary()) # Shows total videos, Real/Fake counts per dataset

# Get 100 perfectly balanced videos across all available datasets handling classes
sample = manager.sample_subset(n=100, balanced=True)
```

---

## 3. Logs and Debugging

The `pipeline_runner.py` script automatically outputs clean, readable alignments to your console, and simultaneously writes everything to `pipeline_run_logs/pipeline_run.log`. If something fails, the error will be printed neatly with a `[FAIL]` prefix.

Example terminal output:
```text
16:42:01  INFO   ==============================================================
16:42:01  INFO     PIPELINE START
16:42:01  INFO   --------------------------------------------------------------
16:42:01  INFO     Total videos available : 11520
...
16:42:15  INFO   [1/5]  id50_0001.mp4
16:42:38  INFO     OK  id50_0001         Celeb-DF-v2  REAL  319 fr   23.4s
```

---

## 4. Testing

The codebase includes a comprehensive test suite to ensure individual components work correctly and the final `.pt` schema precisely matches Phase 1 requirements.

To run all unit and integration tests:
```bash
pytest tests/ -v
```

**Unit Tests (`tests/unit/`)**:
* `test_audio_processor.py`: Verifies fallback zero-padding for silent videos and MFCC/Mel extraction.
* `test_dataset_manager.py`: Tests hierarchical dataset-first sampling and dataset directory discovery.
* `test_face_tracker.py`: Tests the First-Select, Then-Lock IOU MTCNN tracking algorithm logic.
* `test_feature_extractor.py`: Ensures valid bounding box crops and RPPG/Residual tensor alignments.
* `test_video_preprocessor.py`: Tests FFmpeg/OpenCV frame extraction rates and dimension squashing.

**Integration Tests (`tests/integration/`)**:
* `test_pipeline_integration.py`: End-to-end test. Runs real dataset videos continuously through every component.
* `test_output_files.py`: Iterates `test_output/` to guarantee final extracted `.pt` payload geometries perfectly match the expected schema.

*Note: The official extraction logs are stored in `pipeline_run_logs/`. Any raw console text logs from manually triggered experimental runs or testing outputs should be safely archived in the `test_logs/` directory to prevent repository root directory clutter.*
