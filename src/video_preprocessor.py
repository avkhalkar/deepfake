"""
video_preprocessor.py
---------------------
Extracts frames from a video at a fixed frame rate and spatial resolution.

The extractor reads a source video, resamples it to a target FPS (default 25),
resizes every frame to a target resolution (default 1280x720), and converts
the colour space from BGR (OpenCV default) to RGB.

Usage (standalone):
    from video_preprocessor import VideoPreprocessor

    vp = VideoPreprocessor(target_fps=25, target_res=(1280, 720))

    frames, indices, fps = vp.extract_frames("path/to/video.mp4")
    # frames  -> np.ndarray  [T, 720, 1280, 3]  uint8, RGB
    # indices -> list[int]    original frame numbers that were sampled
    # fps     -> float        source video's native frame rate

Usage (CLI quick-test):
    python video_preprocessor.py
"""

import cv2
import numpy as np
import os
from typing import List, Tuple


class VideoPreprocessor:
    """
    Handles frame extraction and spatial normalisation.

    Parameters
    ----------
    target_fps : int
        Desired output frame rate. Frames are sampled uniformly from the
        source timeline so that the output matches this rate.
    target_res : tuple (width, height)
        Every extracted frame is resized to this resolution.
    """

    def __init__(self, target_fps: int = 25, target_res: Tuple[int, int] = (1280, 720)):
        self.target_fps = target_fps
        self.target_res = target_res          # (width, height)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_frames(self, video_path: str) -> Tuple[np.ndarray, List[int], float]:
        """
        Read *video_path* and return uniformly-sampled, resized RGB frames.

        Returns
        -------
        frames : np.ndarray, shape [T, H, W, 3], dtype uint8
        frame_indices : list[int] – which source-frame numbers were kept
        source_fps : float – the video's native FPS
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        source_fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps == 0:
            cap.release()
            raise ValueError(f"Invalid FPS (0) for video: {video_path}")

        # --- decide which frames to keep ---
        duration         = total_frames / source_fps
        num_target_frames = int(duration * self.target_fps)

        if num_target_frames > 0:
            sample_indices = np.linspace(0, total_frames - 1, num_target_frames).astype(int)
            sample_indices = sorted(set(sample_indices))
        else:
            sample_indices = []

        target_set = set(sample_indices)

        # --- read & resize ---
        extracted_frames = []
        actual_indices   = []

        for i in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                break

            if i in target_set:
                resized = cv2.resize(frame, self.target_res)
                rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                extracted_frames.append(rgb)
                actual_indices.append(i)

        cap.release()

        # --- pack into a single array ---
        if extracted_frames:
            frames_array = np.array(extracted_frames, dtype=np.uint8)
        else:
            frames_array = np.empty(
                (0, self.target_res[1], self.target_res[0], 3), dtype=np.uint8
            )

        return frames_array, actual_indices, source_fps


if __name__ == "__main__":
    pass
