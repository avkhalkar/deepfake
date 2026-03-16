"""
audio_processor.py
------------------
Extracts audio features (MFCC and Mel-spectrogram) from a video file.

Only DFDC videos contain audio.  For all other datasets (Celeb-DF,
DeeperForensics, FaceForensics++) this module gracefully returns
zero-valued tensors with ``has_audio=False``.

The module probes each file for an audio stream via ``ffprobe`` before
attempting extraction, so missing ``ffmpeg`` will simply cause silent
fallback — no crash.

Usage:
    from audio_processor import AudioProcessor

    ap = AudioProcessor(sr=16000, n_mfcc=13, n_mels=128)
    result = ap.extract_features("path/to/video.mp4")

    result["mfcc"]             # np.ndarray [n_mfcc, T]   or [n_mfcc, 1] zeros
    result["mel_spectrogram"]  # np.ndarray [n_mels, T]   or [n_mels, 1] zeros
    result["has_audio"]        # bool
"""

import numpy as np
import os
import subprocess
from typing import Tuple


class AudioProcessor:
    """
    MFCC and Mel-spectrogram feature extractor with safe silent-video fallback.

    Parameters
    ----------
    sr : int
        Target sample rate (Hz).  Audio is resampled to this rate.
    n_mfcc : int
        Number of MFCC coefficients to extract.
    n_mels : int
        Number of Mel-frequency bins for the spectrogram.
    hop_length : int
        STFT hop size in samples (controls time resolution).
    """

    def __init__(self, sr: int = 16000, n_mfcc: int = 13,
                 n_mels: int = 128, hop_length: int = 512):
        self.sr         = sr
        self.n_mfcc     = n_mfcc
        self.n_mels     = n_mels
        self.hop_length = hop_length

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _probe_audio(self, video_path: str) -> bool:
        """Return True if *video_path* contains at least one audio stream."""
        try:
            result = subprocess.run(
                ["ffprobe", "-i", video_path, "-show_streams",
                 "-select_streams", "a", "-loglevel", "error"],
                capture_output=True, text=True, timeout=10,
            )
            return len(result.stdout.strip()) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _extract_audio_waveform(self, video_path: str) -> np.ndarray:
        """Load the audio track as a mono waveform"""
        import librosa
        import warnings
        
        # Try loading audio directly from video, suppressing PySoundFile/audioread warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _ = librosa.load(video_path, sr=self.sr, mono=True)
            return y
        except Exception:
            return np.array([])

    def _make_silent_result(self):
        """Return the standard zero-tensor dict for videos without audio."""
        return {
            "mfcc":            np.zeros((self.n_mfcc, 0)),
            "mel_spectrogram": np.zeros((self.n_mels, 0)),
            "has_audio":       False,
        }

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def extract_features(self, video_path: str) -> dict:
        """
        Extract audio features from a video file.

        Returns
        -------
        dict
            mfcc            : np.ndarray [n_mfcc, T] or zeros  
            mel_spectrogram : np.ndarray [n_mels, T] or zeros  
            has_audio       : bool
        """
        # --- guard: file missing ---
        if not os.path.exists(video_path):
            return self._make_silent_result()

        # --- guard: no audio stream ---
        if not self._probe_audio(video_path):
            return self._make_silent_result()

        # --- load waveform ---
        waveform = self._extract_audio_waveform(video_path)
        if len(waveform) == 0:
            return self._make_silent_result()

        # --- compute features ---
        import librosa

        mfcc = librosa.feature.mfcc(
            y=waveform, sr=self.sr,
            n_mfcc=self.n_mfcc, hop_length=self.hop_length,
        )

        mel_spec    = librosa.feature.melspectrogram(
            y=waveform, sr=self.sr,
            n_mels=self.n_mels, hop_length=self.hop_length,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return {
            "mfcc":            mfcc,
            "mel_spectrogram": mel_spec_db,
            "has_audio":       True,
        }
