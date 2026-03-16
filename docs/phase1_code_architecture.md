# Phase 1 Code Architecture & Implementation Strategy

This document outlines how the Phase 1 goals (from [checklist.md](file:///d:/Main/Club%20Projects/AI%20Club%20Projects/DeepFake/deepfake/checklist.md) and [implementation_plan.md](file:///d:/Main/Club%20Projects/AI%20Club%20Projects/DeepFake/implementation_plan.md)) will be implemented in code, specifically handling the diverse dataset structures provided.

## 1. Dataset Handling Strategy (`dataset_manager.py`)
Since we have four different datasets with unique structures and metadata formats, we will implement an Object-Oriented approach using a **Base Class**:

### `BaseDataset` (Abstract Class)
Defines the required methods: `__len__()`, `__getitem__()`, and `get_metadata()`. All dataset-specific classes will inherit from this to ensure the pipeline receives standardized data.

### Implementations:
- **`CelebDFDataset(BaseDataset)`**: Parses `List_of_testing_videos.txt` to map videos in `Celeb-real`, `YouTube-real`, and `Celeb-synthesis`.
- **`DeeperForensicsDataset(BaseDataset)`**: Reads the `lists/` files to identify real (`source_videos_part_03`) vs. manipulated (`manipulated_videos_part_01`).
- **`DFDCDataset(BaseDataset)`**: Parses `train_sample_videos/metadata.json` for labels.
- **`FaceForensicsDataset(BaseDataset)`**: Reads the `csv` files to handle the 5 different categories and map them to their specific manipulation types.

The unified `DatasetManager` will instantiate these classes and yield a standardized dictionary for each video:
`{'video_path': str, 'label': int, 'dataset_name': str, 'manipulation_type': str, 'has_audio': bool}`

## 2. Module Breakdown

The code will be split into modular, object-oriented Python files inside a `src/` directory to keep things clean and maintainable.

### A. [src/video_preprocessor.py](file:///d:/Main/Club%20Projects/AI%20Club%20Projects/DeepFake/deepfake/src/video_preprocessor.py) (In Progress)
- **Goal**: Read `.mp4` files using OpenCV (`cv2`).
- **Functionality**: Extracts frames at exactly 25 FPS and resizes them to 1280x720. 
- *Note: OpenCV can natively read `.mp4` files on Windows without extra extensions.*

### B. `face_tracker.py`
- **Goal**: Implement the "First-Select, Then-Lock" logic.
- **Functionality**: 
  - Runs MTCNN on the frames to get bounding boxes.
  - Initializes a Kalman Filter (using a lightweight SORT implementation) for tracking.
  - Selects the primary subject in the first few frames (based on bounding box size).
  - Locks onto that subject ID and tracks them across all frames, using IOU matching to maintain identity even if multiple faces are present.

### C. `feature_extractor.py`
- **Goal**: Extract spatial, geometric, and physiological features from the locked face track.
- **Functionality**:
  - **Alignment & Landmarks**: Uses `mediapipe` or `dlib` to find 68 landmarks and align the face, cropping it to 224x224.
  - **Pose Angles**: Calculates pitch, yaw, and roll from landmarks.
  - **Temporal Smoothing**: Applies a Savitzky-Golay filter (`scipy.signal.savgol_filter`) to the landmarks and pose angles over time.
  - **Residual Maps**: Applies a high-pass Spatial Rich Model (SRM) filter to the 224x224 face crops to highlight deepfake artifacts.
  - **rPPG**: Extracts average color intensities over specific skin regions across the face track to form the physiological signal.

### D. `audio_processor.py`
- **Goal**: Extract Audio-Visual stream features.
- **Functionality**: 
  - Attempts to separate audio using `librosa`.
  - **Missing Audio Fallback**: If the video has no audio track (e.g., FaceForensics++, Celeb-DF, DeeperForensics), the module will return a zero-tensor of the correct shape `[T, F]` and set a flag (e.g., `has_audio = False`) rather than throwing an exception.
  - If audio exists (DFDC), it computes MFCCs and log Mel-spectrograms aligned to the video's temporal timeframe.

### E. `pipeline_runner.py` (The Main Script)
- **Goal**: Orchestrate the entire flow and manage memory.
- **Functionality**:
  1. Requests a video from `dataset_manager.py`.
  2. Passes it sequentially through the modules.
  3. Formats all outputs into the required PyTorch schema (keys: `video_frames`, `face_crops`, `residual_maps`, etc.).
  4. Saves the resulting dictionary as a `.pt` file using `torch.save()`.

### F. `tests/` Directory
- **Goal**: Ensure the reliability and correctness of the pipeline through automated testing.
- **Unit Tests (`tests/unit/`)**:
  - [test_video_preprocessor.py](file:///d:/Main/Club%20Projects/AI%20Club%20Projects/DeepFake/deepfake/tests/unit/test_video_preprocessor.py): Verify frame extraction counts and resizing logic.
  - `test_face_tracker.py`: Mock MTCNN outputs to verify SORT data association and identity locking logic.
  - `test_feature_extractor.py`: Validate output shapes of spatial and temporal features (e.g., ensuring `face_crops` is exactly `[T, 224, 224, 3]`).
- **Integration Tests (`tests/integration/`)**:
  - `test_pipeline_integration.py`: Run a small, mocked `.mp4` file through the entire pipeline (from `DatasetManager` to `.pt` file creation) to ensure all modules connect and pass data correctly without crashing.

## 3. Execution Plan
To implement this, we will build and test these modules one by one:
1. Finish [src/video_preprocessor.py](file:///d:/Main/Club%20Projects/AI%20Club%20Projects/DeepFake/deepfake/src/video_preprocessor.py) integration.
2. Build and run unit tests for [video_preprocessor.py](file:///d:/Main/Club%20Projects/AI%20Club%20Projects/DeepFake/deepfake/video_preprocessor.py).
3. Build `src/face_tracker.py` and its unit tests.
4. Build `src/feature_extractor.py` and its unit tests.
5. Build `src/audio_processor.py`.
6. Tie it all together with `src/pipeline_runner.py` and `src/dataset_manager.py`.
7. Write and run the integration test over a sample video.
