# Phase 1: Data Pipeline Checklist

## 1. Environment Setup
- [x] Create virtual environment (`.venv`)
- [x] Activate environment
- [x] Create `requirements.txt` with all 8 library categories
- [x] Install dependencies (`pip install -r requirements.txt`)
- [x] Verify GPU support (if applicable) for PyTorch

## 2. Extraction Pipeline Implementation
- [x] **Frame Extraction**: Exactly 25 FPS, 720p resizing. (`src/video_preprocessor.py`)
- [x] **Face Detection**: Integrate MTCNN/RetinaFace.
- [x] **Identity Locking (SORT)**: 
    - [x] Kalman Filter for motion prediction.
    - [x] IOU-based Hungarian matching.
    - [x] "First-Select, Then-Lock" logic for primary subject.
- [x] **Face Alignment**: Bounding-box crop + resize to 224x224. (`src/feature_extractor.py`)
- [x] **Temporal Smoothing**: Savitzky-Golay filter on rPPG signals. (`src/feature_extractor.py`)
- [x] **High-Frequency Residuals**: SRM filtering on face crops. (`src/feature_extractor.py`)
- [x] **Physiological Extraction**: rPPG signal estimation from skin ROI. (`src/feature_extractor.py`)
- [x] **Audio Extractions**: MFCC and Mel-spectrogram with zero-tensor fallback. (`src/audio_processor.py`)
- [x] **Dataset Management**: BaseDataset ABC + 4 handlers + DatasetManager. (`src/dataset_manager.py`)
- [x] **Pipeline Runner**: Full orchestration with .pt output. (`src/pipeline_runner.py`)
## 3. Testing Implementation

### Unit Tests (`tests/unit/`)
- [x] `test_video_preprocessor.py`
- [x] `test_face_tracker.py`
- [x] `test_feature_extractor.py`
- [x] `test_dataset_manager.py`
- [x] `test_audio_processor.py`

### Integration Tests (`tests/integration/`)
- [x] `test_pipeline_integration.py` (End-to-end on real videos, 6 tests)

## 4. Testing Stages (Incremental Data Runs)

### Stage 1: Functional Test (1–5 Videos)
- [x] Run extraction on 3 Real + 2 Fake videos.
- [x] **Goal**: Ensure no crashes/errors.
- [x] **Check**: Verify tensor shapes match schema (e.g., `face_crops` is `[T, 224, 224, 3]`).

### Stage 2: Validation Test (50–100 Videos)
- [x] Run on mini-batch from FaceForensics++ or DFDC.
- [x] **Goal**: Verify modeling accuracy.
- [x] **Manual Check**: Randomly export 10 crops/residuals to `.jpg` to check alignment.
- [x] **Smoothing Check**: Plot pose angles/rPPG for temporal jitter.
- [x] **Tracking Check**: Verify zero "Identity Hopping" in multi-face scenes.

### Stage 3: Stress Test (200+ Videos)
- [x] Run pipeline overnight on a larger subset (e.g., 200+ videos).
- [x] **Goal**: Stability and Throughput.
- [x] **Check**: Monitor RAM for memory leaks (ensure buffers are flushed to disk).

## 5. Final Verification for Phase 2
- [x] Ensure all 16 schema fields are present and valid in `.pt` outputs.
- [x] Confirm dataset directory structure matches the plan.
