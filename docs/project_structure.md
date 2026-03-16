# Project Directory Structure (Phase 1)

This is the complete directory layout for the `deepfake/` folder, including your existing datasets and the files I will be adding during implementation.

```text
deepfake/
├── datasets/                   # [EXISTING] Raw video datasets
│   ├── Celeb-DF-v2/
│   ├── deeperforensics/
│   ├── dfdc/
│   └── face_forensics/
├── intermediate_outputs/       # [NEW] Temporary storage for pipeline stages
│   ├── extracted_frames/       # 720p JPGs
│   ├── processed_faces/        # 224x224 aligned crops
│   └── audio_tracks/           # Extracted WAV/MP3 files
├── outputs/                    # [NEW] Final packaged tensor files (.pt)
├── tests/                      # [NEW] Automated testing suite
│   ├── unit/                   # Modular unit tests
│   │   ├── test_video_preprocessor.py
│   │   ├── test_face_tracker.py
│   │   ├── test_feature_extractor.py
│   │   └── test_dataset_manager.py
│   └── integration/            # End-to-end pipeline tests
│       └── test_pipeline_integration.py
├── src/                        # [NEW] Main source code modules
│   ├── video_preprocessor.py   # [STARTED] Frame extraction & resizing
│   ├── face_tracker.py         # [NEW] SORT tracker & Identity Locking
│   ├── feature_extractor.py    # [NEW] Landscapes, residuals, rPPG, smoothing
│   ├── audio_processor.py      # [NEW] MFCC & Spectrogram extraction
│   ├── dataset_manager.py      # [NEW] Unified loader for FF++, DFDC, etc.
│   └── pipeline_runner.py      # [NEW] Main orchestration script
├── requirements.txt            # [EXISTING] Environment dependencies
├── pytest.ini                  # [NEW] Pytest configuration for src/ path
├── checklist.md                # [EXISTING] Progress tracking
└── dataset_specification.md    # [EXISTING] Dataset subset manifest
```

### Purpose of New Directories:
- **`intermediate_outputs/`**: Prevents RAM overload by flushing files to disk during processing. These can be cleared after a successful `.pt` export.
- **`outputs/`**: This is where your Phase 2 model will look for training data.
- **`tests/`**: Vital for ensuring the math (alignment, filtering) stays correct as the pipeline grows.
