"""
face_tracker.py
---------------
Detects, tracks, and locks on to the primary face across a video sequence.

Pipeline:
    1. MTCNN detects faces in every frame.
    2. A lightweight Kalman filter (SORT-style) predicts bounding-box motion.
    3. The Hungarian algorithm matches detections to existing tracks via IOU.
    4. "First-Select, Then-Lock" logic picks the largest face as
       the primary subject and follows it even through brief occlusions.

Usage:
    from face_tracker import FaceTracker

    tracker = FaceTracker(device="cuda")     # or "cpu"
    result  = tracker.process_frames(frames) # frames: np.ndarray [T, H, W, 3]

    boxes = result["primary_subject_boxes"]  # np.ndarray [T, 4]  (x1, y1, x2, y2)
    # NaN rows mean the face was lost in that frame.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from facenet_pytorch import MTCNN
from scipy.optimize import linear_sum_assignment


# ======================================================================
#  Utility: Intersection-over-Union
# ======================================================================
def iou(box_a, box_b):
    """
    Compute the Intersection-over-Union between two bounding boxes.

    Each box is [x1, y1, x2, y2].
    """
    xx1 = np.maximum(box_a[0], box_b[0])
    yy1 = np.maximum(box_a[1], box_b[1])
    xx2 = np.minimum(box_a[2], box_b[2])
    yy2 = np.minimum(box_a[3], box_b[3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / (area_a + area_b - inter_area)


# ======================================================================
#  Kalman-based Bounding-Box Tracker (SORT variant)
# ======================================================================
class KalmanBoxTracker:
    """
    Tracks a single bounding box with a constant-velocity Kalman filter.

    Internal state vector:  [cx, cy, scale, ratio, d_cx, d_cy, d_scale]
        cx, cy  = box centre
        scale   = area  (w * h)
        ratio   = aspect ratio  (w / h)
        d_*     = first-order velocity terms
    """

    count = 0                     # class-level ID counter

    def __init__(self, bbox):
        """Initialise a new tracker from an observed [x1, y1, x2, y2] box."""
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # --- state vector (7 x 1) ---
        self.state = np.zeros((7, 1))
        self.state[:4, 0] = self.convert_bbox_to_z(bbox).reshape(-1)

        # --- constant-velocity transition matrix ---
        self.F = np.eye(7)
        self.F[0, 4] = 1   # cx  += d_cx
        self.F[1, 5] = 1   # cy  += d_cy
        self.F[2, 6] = 1   # s   += d_s

        # --- observation matrix (we observe [cx, cy, s, r]) ---
        self.H = np.zeros((4, 7))
        np.fill_diagonal(self.H, 1)              # top-left 4x4 identity

        # --- noise covariances ---
        self.R = np.eye(4) * 10.0                 # measurement noise
        self.Q = np.eye(7) * 0.01                 # process noise
        self.Q[4:, 4:] *= 0.01                    # lower noise on velocities

        self.P = np.eye(7) * 10.0                 # estimate covariance
        self.P[4:, 4:] *= 1000.0                  # high uncertainty on initial velocities

        self.time_since_update = 0
        self.hits = 0

    # ----- Kalman update (measurement received) --------------------------
    def update(self, bbox):
        """Correct the state with a new observed bounding box."""
        self.time_since_update = 0
        self.hits += 1

        z = self.convert_bbox_to_z(bbox)
        y = z - self.H @ self.state               # innovation
        S = self.H @ self.P @ self.H.T + self.R   # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    # ----- Kalman predict (time step) ------------------------------------
    def predict(self):
        """Advance the state by one time step; return predicted box."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.state)

    # ----- Coordinate conversions ----------------------------------------
    @staticmethod
    def convert_bbox_to_z(bbox):
        """[x1, y1, x2, y2]  -->  [cx, cy, scale, ratio]  (4x1 column)."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([cx, cy, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """[cx, cy, scale, ratio, ...]  -->  [x1, y1, x2, y2]  (1x4 row)."""
        # Protect against negative values if Kalman filter predicts negative area
        val = max(0.0, float(x[2] * x[3]))
        w = np.sqrt(val)
        h = x[2] / w if w > 0 else 0.0
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# ======================================================================
#  FaceTracker  –  First-Select, Then-Lock
# ======================================================================
class FaceTracker:
    """
    Multi-face detector + single-subject tracker.

    Parameters
    ----------
    device : str or None
        "cuda" / "cpu".  Auto-detected when None.
    max_age : int
        Frames a track survives without a detection before deletion.
    min_hits : int
        Minimum detections before a track is considered confirmed.
    iou_threshold : float
        Minimum IOU to accept a detection-to-track match.
    """

    def __init__(self, device: str = None, max_age: int = 5,
                 min_hits: int = 3, iou_threshold: float = 0.3):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mtcnn = MTCNN(keep_all=True, device=device)

        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

        self.primary_track_id: Optional[int] = None

    # ----- Hungarian assignment ------------------------------------------
    def _associate_detections_to_trackers(self, detections, trackers):
        """
        Match detections to existing trackers using IOU + Hungarian algorithm.

        Returns (matches, unmatched_detections, unmatched_trackers).
        """
        if len(trackers) == 0:
            return (np.empty((0, 2), dtype=int),
                    np.arange(len(detections)),
                    np.empty((0, 5), dtype=int))

        # build IOU cost matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)

        # solve assignment (maximise IOU = minimise negative IOU)
        row_idx, col_idx = linear_sum_assignment(-iou_matrix)
        matched_indices  = np.column_stack((row_idx, col_idx))

        unmatched_dets = [d for d in range(len(detections))  if d not in matched_indices[:, 0]]
        unmatched_trks = [t for t in range(len(trackers))    if t not in matched_indices[:, 1]]

        # reject matches whose IOU is below threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

    # ----- Main entry point ----------------------------------------------
    def process_frames(self, frames: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run detection + tracking on an array of RGB frames.

        Parameters
        ----------
        frames : np.ndarray, shape [T, H, W, 3]

        Returns
        -------
        dict with key "primary_subject_boxes" -> np.ndarray [T, 4]
            Each row is [x1, y1, x2, y2].  Rows of NaN indicate lost frames.
        """
        results = []

        # reset tracker state for this video
        KalmanBoxTracker.count = 0
        self.trackers = []
        self.primary_track_id = None

        for frame in frames:

            # ---- detect faces ----
            boxes, _probs = self.mtcnn.detect(frame)
            if boxes is None:
                boxes = np.empty((0, 4))

            # ---- predict existing trackers forward ----
            predicted = np.zeros((len(self.trackers), 5))
            to_del = []
            for t, trk in enumerate(self.trackers):
                pos = trk.predict()[0]
                predicted[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)

            for t in reversed(to_del):
                self.trackers.pop(t)
                predicted = np.delete(predicted, t, 0)

            # ---- match detections to tracks ----
            matched, unmatched_dets, _ = \
                self._associate_detections_to_trackers(boxes, predicted)

            for m in matched:
                self.trackers[m[1]].update(boxes[m[0]])

            # ---- spawn new tracks for unmatched detections ----
            for i in unmatched_dets:
                self.trackers.append(KalmanBoxTracker(boxes[i]))

            # ---- remove dead tracks ----
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)

            # ---- identity locking ----
            valid = [t for t in self.trackers if t.time_since_update == 0]

            # first frame with faces -> lock on to the largest one
            if self.primary_track_id is None and valid:
                best_area, best_id = 0, None
                for t in valid:
                    box = t.convert_x_to_bbox(t.state)[0]
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > best_area:
                        best_area = area
                        best_id   = t.id
                self.primary_track_id = best_id

            # find the locked subject's box in this frame
            primary_box = None
            if self.primary_track_id is not None:
                for t in valid:
                    if t.id == self.primary_track_id:
                        primary_box = t.convert_x_to_bbox(t.state)[0]
                        break

            # fall back to Kalman prediction during brief occlusion
            if primary_box is None and self.primary_track_id is not None:
                for t in self.trackers:
                    if t.id == self.primary_track_id and t.time_since_update <= self.max_age:
                        primary_box = t.convert_x_to_bbox(t.state)[0]
                        break

            results.append(
                primary_box if primary_box is not None
                else np.array([np.nan, np.nan, np.nan, np.nan])
            )

        return {"primary_subject_boxes": np.array(results)}
