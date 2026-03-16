"""
feature_extractor.py
--------------------
Extracts per-frame spatial and temporal features from a tracked face sequence.

Features produced:
    1. Face crops      - bounding-box region resized to 224x224.
    2. SRM residuals   - high-pass Spatial Rich Model filter that amplifies
                         manipulation artefacts invisible to the naked eye.
                         Output is single-channel Grayscale [T, 224, 224, 1].
    3. rPPG signals    - average skin-colour change per frame (Remote
                         Photoplethysmography), smoothed with a Savitzky-Golay
                         filter to reduce temporal noise.
    4. Landmarks       - 68-point 2D facial landmarks via dlib's shape
                         predictor, output as [T, 68, 2].
    5. Pose angles     - 3D head pose (pitch, yaw, roll) estimated via
                         cv2.solvePnP from the 68 landmarks.

Usage:
    from feature_extractor import FeatureExtractor

    ext = FeatureExtractor(target_size=(224, 224))
    out = ext.process_sequence(frames, bboxes)

    out["face_crops"]      # np.ndarray [T, 224, 224, 3]
    out["residual_maps"]   # np.ndarray [T, 224, 224, 1]
    out["rppg_signals"]    # np.ndarray [T, 3]   (smoothed R, G, B averages)
    out["landmarks"]       # np.ndarray [T, 68, 2]
    out["pose_angles"]     # np.ndarray [T, 3]   (pitch, yaw, roll in degrees)
"""

import numpy as np
import cv2
import dlib
from pathlib import Path
from scipy.signal import savgol_filter


# ======================================================================
#  SRM high-pass kernel
# ======================================================================
def get_srm_filter():
    """Return a 3x3 Spatial Rich Model high-pass kernel."""
    kernel = np.array([[-1, -2, -1],
                       [-2, 12, -2],
                       [-1, -2, -1]], dtype=np.float32) / 4.0
    return kernel


# ======================================================================
#  Canonical 3D face model for solvePnP
# ======================================================================
# 6 canonical 3D points (nose tip, chin, left/right eye corners,
# left/right mouth corners) in a generic coordinate system.
# These are standard anthropometric approximations used widely
# in head-pose estimation literature.
MODEL_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),      # Nose tip           (landmark 30)
    (0.0,   -330.0, -65.0),     # Chin                (landmark 8)
    (-225.0, 170.0, -135.0),    # Left eye left corner (landmark 36)
    (225.0,  170.0, -135.0),    # Right eye right corner (landmark 45)
    (-150.0, -150.0, -125.0),   # Left mouth corner   (landmark 48)
    (150.0,  -150.0, -125.0),   # Right mouth corner  (landmark 54)
], dtype=np.float64)

# Indices into the 68-landmark array that correspond to the 6 points above
POSE_LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]


# ======================================================================
#  FeatureExtractor
# ======================================================================
class FeatureExtractor:
    """
    Spatial + temporal feature extraction for a tracked face.

    Parameters
    ----------
    target_size : tuple (width, height)
        All face crops are resized to this size before feature computation.
    landmark_model_path : str or None
        Path to dlib's 68-point shape predictor .dat file.
        If None, auto-resolves to ``models/shape_predictor_68_face_landmarks.dat``
        relative to the project root (one level above ``src/``).
    """

    def __init__(self, target_size=(224, 224), landmark_model_path=None):
        self.target_size = target_size
        self.srm_kernel  = get_srm_filter()

        # --- dlib landmark predictor ---
        if landmark_model_path is None:
            # Resolve relative to project root: src/../models/
            project_root = Path(__file__).resolve().parent.parent
            landmark_model_path = str(project_root / "models" / "shape_predictor_68_face_landmarks.dat")

        if not Path(landmark_model_path).exists():
            raise FileNotFoundError(
                f"dlib shape predictor not found at: {landmark_model_path}\n"
                "Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )

        self.landmark_predictor = dlib.shape_predictor(landmark_model_path)
        # dlib face detector used as a fallback for landmark region
        # We use a full-image rectangle since our crops are already tightly framed
        self._full_rect = None  # lazily set per target_size

    # ------------------------------------------------------------------
    #  SRM residual maps (1-channel Grayscale)
    # ------------------------------------------------------------------
    def apply_srm(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the SRM high-pass filter to amplify compression artefacts.
        Returns a uint8 image with shape (H, W, 1) in Grayscale.
        """
        # Convert to Grayscale (Luminance)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Apply high-pass filter
        filtered = cv2.filter2D(gray, -1, self.srm_kernel)

        # Clip to valid range and cast to uint8
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        # Expand dims to explicitly match schema [H, W, 1] instead of [H, W]
        return np.expand_dims(filtered, axis=-1)

    # ------------------------------------------------------------------
    #  Face cropping
    # ------------------------------------------------------------------
    def get_face_crops(self, frames: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Crop and resize the face region from every frame.

        If the bounding box is NaN (tracking lost), a black frame is used.
        """
        crops = []
        for frame, bbox in zip(frames, bboxes):

            # tracking lost -> black placeholder
            if np.any(np.isnan(bbox)):
                crops.append(np.zeros((*self.target_size[::-1], 3), dtype=np.uint8))
                continue

            x1, y1, x2, y2 = map(int, bbox[:4])
            h, w = frame.shape[:2]

            # clamp to frame boundaries
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((*self.target_size[::-1], 3), dtype=np.uint8))
            else:
                crop = frame[y1:y2, x1:x2]
                crops.append(cv2.resize(crop, self.target_size))

        return np.array(crops)

    # ------------------------------------------------------------------
    #  68-point facial landmarks (dlib)
    # ------------------------------------------------------------------
    def extract_landmarks(self, face_crops: np.ndarray) -> np.ndarray:
        """
        Extract 68 facial landmark (x, y) coordinates from each face crop
        using dlib's shape predictor.

        The predictor runs on each 224x224 RGB crop directly, using a
        full-image bounding rectangle (since the crop is already tightly
        framed around the face).

        Returns shape [T, 68, 2].
        """
        h, w = self.target_size[::-1]  # height, width

        # Create a dlib rectangle spanning the full crop image
        if self._full_rect is None:
            self._full_rect = dlib.rectangle(0, 0, self.target_size[0], self.target_size[1])

        all_landmarks = []
        for crop in face_crops:
            # Check for black/zero frames (tracking lost)
            if crop.max() == 0:
                all_landmarks.append(np.zeros((68, 2), dtype=np.float32))
                continue

            # Convert to grayscale for dlib
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

            # Predict 68 landmarks
            shape = self.landmark_predictor(gray, self._full_rect)

            # Convert dlib shape to numpy array [68, 2]
            points = np.array(
                [(shape.part(i).x, shape.part(i).y) for i in range(68)],
                dtype=np.float32
            )
            all_landmarks.append(points)

        return np.array(all_landmarks)

    # ------------------------------------------------------------------
    #  3D Head Pose estimation (Pitch, Yaw, Roll)
    # ------------------------------------------------------------------
    def estimate_pose(self, landmarks: np.ndarray, face_crops: np.ndarray) -> np.ndarray:
        """
        Estimate head pose (pitch, yaw, roll) in degrees from the 68
        facial landmarks using cv2.solvePnP.

        Uses a canonical 3D face model and a simple pinhole camera model
        based on the crop dimensions.

        Returns shape [T, 3] with angles in degrees.
        """
        h, w = self.target_size[::-1]  # height, width

        # Approximate camera intrinsics (pinhole model)
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1.0],
        ], dtype=np.float64)

        # No lens distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        all_angles = []
        for lm in landmarks:
            # Check for zero landmarks (tracking lost)
            if np.all(lm == 0):
                all_angles.append(np.zeros(3, dtype=np.float32))
                continue

            # Extract the 6 key landmark points for PnP
            image_points = lm[POSE_LANDMARK_INDICES].astype(np.float64)

            # Solve PnP to get rotation and translation vectors
            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                all_angles.append(np.zeros(3, dtype=np.float32))
                continue

            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # Decompose rotation matrix into Euler angles
            # Using cv2.decomposeProjectionMatrix for clean extraction
            proj_matrix = np.hstack((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

            # euler_angles are [pitch, yaw, roll] in degrees
            pitch = euler_angles[0, 0]
            yaw   = euler_angles[1, 0]
            roll  = euler_angles[2, 0]

            all_angles.append(np.array([pitch, yaw, roll], dtype=np.float32))

        return np.array(all_angles)

    # ------------------------------------------------------------------
    #  rPPG estimation (simplified skin-colour averaging)
    # ------------------------------------------------------------------
    def extract_rppg(self, face_crops: np.ndarray) -> np.ndarray:
        """
        Estimate a raw rPPG signal by averaging RGB values over the central
        50 % of each face crop (roughly cheeks + nose area).

        Returns shape [T, 3]  (one RGB triplet per frame).
        """
        h, w = self.target_size
        y0, y1 = int(h * 0.3), int(h * 0.8)
        x0, x1 = int(w * 0.3), int(w * 0.7)

        signals = []
        for crop in face_crops:
            roi = crop[y0:y1, x0:x1]
            signals.append(roi.mean(axis=(0, 1)))      # [R, G, B] averages

        return np.array(signals)

    # ------------------------------------------------------------------
    #  Temporal smoothing (Savitzky-Golay)
    # ------------------------------------------------------------------
    def smooth_signals(self, signals: np.ndarray,
                       window_length: int = 5, polyorder: int = 2) -> np.ndarray:
        """
        Apply a Savitzky-Golay filter along the time axis.

        If the sequence is shorter than *window_length*, returns unsmoothed.
        """
        if len(signals) < window_length:
            return signals
        return savgol_filter(signals, window_length=window_length,
                             polyorder=polyorder, axis=0)

    # ------------------------------------------------------------------
    #  Full sequence pipeline
    # ------------------------------------------------------------------
    def process_sequence(self, frames: np.ndarray, bboxes: np.ndarray) -> dict:
        """
        Run the complete feature extraction pipeline on a tracked sequence.

        Returns a dict of numpy arrays (see module docstring for shapes).
        """
        face_crops    = self.get_face_crops(frames, bboxes)
        residual_maps = np.array([self.apply_srm(crop) for crop in face_crops])
        raw_rppg      = self.extract_rppg(face_crops)
        smoothed_rppg = self.smooth_signals(raw_rppg)

        # 68-point landmarks from dlib
        landmarks = self.extract_landmarks(face_crops)

        # 3D head pose from landmarks
        pose_angles = self.estimate_pose(landmarks, face_crops)

        return {
            "face_crops":    face_crops,
            "residual_maps": residual_maps,
            "rppg_signals":  smoothed_rppg,
            "landmarks":     landmarks,
            "pose_angles":   pose_angles,
        }
