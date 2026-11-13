"""Shared MediaPipe pose utilities.

Every stage of the pipeline represents a body pose as a flat vector of
33 landmarks x (x, y, z) = 99 features, in MediaPipe landmark order.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

NUM_LANDMARKS = 33

# Official Google-hosted PoseLandmarker model (heavy variant, float16).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
DEFAULT_TASK_PATH = Path("models") / "pose_landmarker_heavy.task"

# Form-quality classes and the keys used to label them during collection.
CLASS_KEY_MAP = {
    "u": "Up",
    "d": "Down",
    "o": "Optimal",
    "s": "subOptimal",
    "x": "Dangerous",
}

FEATURE_COLUMNS = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

CSV_HEADERS = ["label"] + FEATURE_COLUMNS

# MediaPipe Pose landmark pairs used for the live skeleton overlay.
POSE_CONNECTIONS = (
    (0, 11), (0, 12),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)


def ensure_task_model(path: str | Path = DEFAULT_TASK_PATH) -> Path:
    """Return the path to the PoseLandmarker .task model, downloading it if missing."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PoseLandmarker model to {path} (~30 MB, one-time)...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Download complete.")
    return path


def create_landmarker(task_path: str | Path = DEFAULT_TASK_PATH):
    """Create a synchronous (VIDEO-mode) MediaPipe PoseLandmarker.

    Use as a context manager:
        with create_landmarker() as landmarker: ...
    """
    import mediapipe as mp

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(ensure_task_model(task_path))
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def detect(landmarker, frame_bgr, timestamp_ms: int):
    """Run pose detection on a BGR OpenCV frame."""
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def result_to_row(result) -> list[float] | None:
    """Flatten the first detected pose into a 99-value feature row, or None."""
    if not result.pose_landmarks:
        return None
    row: list[float] = []
    for lm in result.pose_landmarks[0]:
        row.extend([lm.x, lm.y, lm.z])
    return row


def canonicalize_row(row) -> list[float]:
    """Express landmarks relative to the hip center and torso scale/orientation."""
    import numpy as np

    points = np.asarray(row, dtype=float).reshape(NUM_LANDMARKS, 3).copy()
    hip_center = points[[23, 24]].mean(axis=0)
    shoulder_center = points[[11, 12]].mean(axis=0)
    torso_xy = shoulder_center[:2] - hip_center[:2]
    torso_length = float(np.linalg.norm(torso_xy))
    if torso_length < 1e-6:
        return points.reshape(-1).tolist()

    angle = -np.pi / 2 - np.arctan2(torso_xy[1], torso_xy[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    points[:, :2] = (points[:, :2] - hip_center[:2]) @ rotation.T / torso_length
    points[:, 2] = (points[:, 2] - hip_center[2]) / torso_length
    return points.reshape(-1).tolist()


def draw_landmarks(frame_bgr, result, color=(95, 205, 245)) -> None:
    """Draw the translucent 3D-style mannequin over the camera frame."""
    from formcoach.mannequin import draw_mannequin

    if not result.pose_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    landmarks = result.pose_landmarks[0]
    points = [(lm.x * w, lm.y * h) for lm in landmarks]
    visible = [float(getattr(lm, "visibility", 1.0) or 0.0) >= 0.35 for lm in landmarks]
    draw_mannequin(frame_bgr, points, color=color, visibility=visible)
