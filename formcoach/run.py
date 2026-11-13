"""Real-time form feedback: webcam -> pose -> classifier -> Arduino.

Streams webcam frames through MediaPipe pose estimation, classifies form
quality with the trained model for the active exercise, and sends the
predicted class over serial to the Arduino, which drives the RGB LED,
buzzer, and servo.

Serial protocol: lowercase class name + newline (e.g. "dangerous\\n"),
sent only when the prediction changes and confidence >= --threshold.

Keys: 1-9 switch exercise, q quits.
Press t to calibrate with a held T-pose, then enable the person-aligned coach;
press t again to turn the coach off.

Usage:
    python -m formcoach.run --port COM7
    python -m formcoach.run --no-serial            # software-only, no Arduino
    python -m formcoach.run --camera 1 --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from formcoach import pose
from formcoach.mannequin import draw_mannequin


class PredictionSmoother:
    """Average recent frame probabilities and debounce label changes."""

    def __init__(self, window: int = 7, confirm_frames: int = 3):
        self.probabilities: deque[np.ndarray] = deque(maxlen=max(1, window))
        self.confirm_frames = max(1, confirm_frames)
        self.stable_label: str | None = None
        self.candidate_label: str | None = None
        self.candidate_count = 0

    def reset(self) -> None:
        self.probabilities.clear()
        self.stable_label = None
        self.candidate_label = None
        self.candidate_count = 0

    def update(self, classes, probabilities: np.ndarray) -> tuple[str, float]:
        self.probabilities.append(np.asarray(probabilities, dtype=float))
        averaged = np.mean(self.probabilities, axis=0)
        index = int(np.argmax(averaged))
        candidate = str(classes[index])
        if candidate == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = candidate
            self.candidate_count = 1

        if self.stable_label is None or self.candidate_count >= self.confirm_frames:
            self.stable_label = candidate
        stable_index = list(classes).index(self.stable_label)
        return self.stable_label, float(averaged[stable_index])


class TposeCalibrator:
    """Collect a short, stable T-pose and capture the user's body proportions."""

    REQUIRED_FRAMES = 15
    REQUIRED_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

    def __init__(self):
        self.samples: deque[np.ndarray] = deque(maxlen=self.REQUIRED_FRAMES)

    def reset(self) -> None:
        self.samples.clear()

    @staticmethod
    def _angle(a, b, c) -> float:
        first = a - b
        second = c - b
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denom < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def update(self, result, row, frame_shape) -> tuple[bool, int, list[float] | None]:
        if row is None or not result.pose_landmarks:
            self.reset()
            return False, 0, None

        landmarks = result.pose_landmarks[0]
        if any(float(getattr(landmarks[i], "visibility", 1.0) or 0.0) < 0.55
               for i in self.REQUIRED_LANDMARKS):
            self.reset()
            return False, 0, None

        frame_height, frame_width = frame_shape[:2]
        points = np.asarray(
            [(landmarks[i].x * frame_width, landmarks[i].y * frame_height)
             for i in range(pose.NUM_LANDMARKS)],
            dtype=float,
        )
        shoulders = (points[11] + points[12]) / 2
        hips = (points[23] + points[24]) / 2
        torso_length = float(np.linalg.norm(shoulders - hips))
        shoulder_width = float(np.linalg.norm(points[11] - points[12]))
        if torso_length < 20 or shoulder_width < 10:
            self.reset()
            return False, 0, None

        arms_level = max(
            abs(points[i, 1] - shoulders[1]) / torso_length
            for i in (13, 14, 15, 16)
        ) < 0.28
        arms_wide = abs(points[15, 0] - points[16, 0]) > shoulder_width * 1.65
        torso_upright = abs(shoulders[0] - hips[0]) / torso_length < 0.38
        elbows_straight = (
            self._angle(points[11], points[13], points[15]) > 145
            and self._angle(points[12], points[14], points[16]) > 145
        )
        valid = arms_level and arms_wide and torso_upright and elbows_straight
        if not valid:
            self.reset()
            return False, 0, None

        current = np.asarray(row, dtype=float)
        if self.samples:
            previous_norm = np.asarray(pose.canonicalize_row(self.samples[-1])).reshape(
                pose.NUM_LANDMARKS, 3
            )
            current_norm = np.asarray(pose.canonicalize_row(current)).reshape(
                pose.NUM_LANDMARKS, 3
            )
            tracked = np.asarray((11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28))
            drift = float(np.mean(np.linalg.norm(
                current_norm[tracked, :2] - previous_norm[tracked, :2], axis=1
            )))
            if drift > 0.04:
                self.reset()
        self.samples.append(current)
        progress = len(self.samples)
        if progress < self.REQUIRED_FRAMES:
            return True, progress, None
        calibrated = np.mean(np.stack(self.samples), axis=0).tolist()
        self.reset()
        return True, progress, calibrated


# MediaPipe landmark groups used to size and bound the coaching mannequins.
POSE_CONNECTIONS = [
    (0, 11), (0, 12),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def load_reference_poses(data_dir: str | Path, exercise: str, feature_columns: list[str]):
    """Choose representative Up, Optimal, and Down samples for an exercise."""
    csv_path = Path(data_dir) / f"{exercise}.csv"
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        return {}

    references = {}
    labels = df["label"].astype(str).str.casefold()
    for label in ("Up", "Optimal", "Down"):
        samples = df.loc[labels == label.casefold(), feature_columns]
        if samples.empty:
            continue
        values = samples.to_numpy(dtype=float)
        center = values.mean(axis=0)
        references[label] = values[int(np.square(values - center).sum(axis=1).argmin())].tolist()
    return references


def load_exercise_names(path: str | Path) -> dict[str, str]:
    """Load display labels while keeping dataset/model keys unchanged."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {str(key): str(value) for key, value in payload.items()}


def make_reference_panel(frame_shape, exercise: str, reference_poses, width: int):
    """Render Up, Optimal, and Down landmark references beside the camera."""
    import cv2

    height, frame_width = frame_shape[:2]
    panel = np.full((height, width, 3), (28, 32, 38), dtype=np.uint8)
    cv2.putText(panel, "MOVEMENT REFERENCES", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (100, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, exercise, (16, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (235, 235, 235), 1, cv2.LINE_AA)

    labels = ("Up", "Optimal", "Down")
    label_colors = {"Up": (100, 230, 255), "Optimal": (95, 205, 150), "Down": (255, 180, 100)}
    top, bottom, gap = 82, height - 18, 12
    cell_width = (width - 2 * 16 - 2 * gap) // len(labels)
    cell_height = bottom - top
    margin = 10

    for index, label in enumerate(labels):
        left = 16 + index * (cell_width + gap)
        right = left + cell_width
        cv2.rectangle(panel, (left, top), (right, bottom), (38, 43, 50), -1)
        title = "SAFE" if label == "Optimal" else label.upper()
        cv2.putText(panel, title, (left + 8, top + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, label_colors[label], 2, cv2.LINE_AA)

        reference = reference_poses.get(label)
        if reference is None:
            cv2.putText(panel, "No sample", (left + 8, top + 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (180, 180, 180), 1, cv2.LINE_AA)
            continue

        # Rotate 90 degrees clockwise, then fit the swapped image dimensions
        # into this pose card without stretching the figure.
        rotated_width = height
        rotated_height = frame_width
        available_width = cell_width - 2 * margin
        available_height = cell_height - 42 - margin
        scale = min(available_width / rotated_width, available_height / rotated_height)
        drawn_width = rotated_width * scale
        drawn_height = rotated_height * scale
        offset_x = left + (cell_width - drawn_width) / 2
        offset_y = top + 32 + (cell_height - 32 - drawn_height) / 2

        points = []
        for i in range(pose.NUM_LANDMARKS):
            x, y = reference[i * 3], reference[i * 3 + 1]
            rotated_x = height - y * height
            rotated_y = x * frame_width
            points.append((int(offset_x + rotated_x * scale),
                           int(offset_y + rotated_y * scale)))

        draw_mannequin(panel, points, color=(95, 205, 150), alpha=0.78,
                       width_scale=max(0.55, scale * 1.7))

    return panel


def make_calibration_panel(frame_shape, display_name: str, width: int,
                           progress: int, required_frames: int, pose_valid: bool):
    """Show an upright T-pose example and calibration instructions."""
    import cv2

    height, _ = frame_shape[:2]
    panel = np.full((height, width, 3), (28, 32, 38), dtype=np.uint8)
    cv2.putText(panel, "T-POSE CALIBRATION", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (100, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, display_name[:34], (16, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (235, 235, 235), 1, cv2.LINE_AA)

    # Upright mannequin guide: arms level with the shoulders, legs straight.
    figure = np.zeros((pose.NUM_LANDMARKS, 2), dtype=float)
    figure_points = {
        0: (width // 2, 132), 11: (width // 2 - 24, 142), 12: (width // 2 + 24, 142),
        13: (width // 2 - 90, 142), 15: (width // 2 - 164, 142),
        14: (width // 2 + 90, 142), 16: (width // 2 + 164, 142),
        23: (width // 2 - 20, 225), 24: (width // 2 + 20, 225),
        25: (width // 2 - 22, 295), 27: (width // 2 - 24, 365),
        26: (width // 2 + 22, 295), 28: (width // 2 + 24, 365),
    }
    for index, point in figure_points.items():
        figure[index] = point
    visible = np.zeros(pose.NUM_LANDMARKS, dtype=bool)
    visible[list(figure_points)] = True
    draw_mannequin(panel, figure, color=(95, 205, 150), alpha=0.82,
                   visibility=visible, width_scale=1.5)

    instruction = "Face camera; full body in view; arms straight and level"
    cv2.putText(panel, instruction, (16, height - 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (205, 210, 220), 1, cv2.LINE_AA)
    status = (f"Hold steady: {progress}/{required_frames} frames" if pose_valid
              else "Move into the shown T-pose")
    status_color = (95, 205, 150) if pose_valid else (100, 200, 255)
    cv2.putText(panel, status, (16, height - 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, status_color, 1, cv2.LINE_AA)
    bar_left, bar_top, bar_right, bar_bottom = 16, height - 32, width - 16, height - 18
    cv2.rectangle(panel, (bar_left, bar_top), (bar_right, bar_bottom), (70, 75, 82), 1)
    fraction = min(1.0, progress / max(1, required_frames))
    cv2.rectangle(panel, (bar_left + 1, bar_top + 1),
                  (bar_left + int((bar_right - bar_left - 1) * fraction), bar_bottom - 1),
                  (95, 205, 150) if pose_valid else (70, 75, 82), -1)
    return panel


def level_target_and_live(target: np.ndarray, live: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Level the target's ankle line without changing target/live alignment."""
    ankle_axis = target[28] - target[27]
    ankle_length = float(np.linalg.norm(ankle_axis))
    if ankle_length < 1e-6:
        return target, live

    # An ankle-to-ankle line has no direction. Normalize its angle to
    # [-90°, 90°] so reversed landmark order never flips the whole figure.
    line_angle = np.arctan2(ankle_axis[1], ankle_axis[0])
    line_angle = (line_angle + np.pi / 2) % np.pi - np.pi / 2
    angle = -line_angle
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    ground_center = (target[27] + target[28]) / 2
    target = (target - ground_center) @ rotation.T + ground_center
    live = (live - ground_center) @ rotation.T + ground_center
    return target, live


def align_reference_to_person(reference_row, live_row, frame_shape) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fit the reference to the live torso's position, scale, and orientation."""
    reference = np.asarray(reference_row, dtype=float).reshape(pose.NUM_LANDMARKS, 3)[:, :2].copy()
    live = np.asarray(live_row, dtype=float).reshape(pose.NUM_LANDMARKS, 3)[:, :2].copy()
    frame_height, frame_width = frame_shape[:2]
    image_scale = np.array([frame_width, frame_height], dtype=float)
    reference *= image_scale
    live *= image_scale

    shoulder_ids = [11, 12]
    hip_ids = [23, 24]
    ref_shoulder = reference[shoulder_ids].mean(axis=0)
    ref_hip = reference[hip_ids].mean(axis=0)
    live_shoulder = live[shoulder_ids].mean(axis=0)
    live_hip = live[hip_ids].mean(axis=0)

    live_axis = live_shoulder - live_hip
    ref_length = float(np.linalg.norm(ref_shoulder - ref_hip))
    live_length = float(np.linalg.norm(live_axis))
    if ref_length < 1e-6 or live_length < 1e-6:
        return None

    ref_axis = ref_shoulder - ref_hip
    angle = np.arctan2(live_axis[1], live_axis[0]) - np.arctan2(ref_axis[1], ref_axis[0])
    scale = live_length / ref_length
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    aligned = (reference - ref_hip) @ rotation.T * scale + live_hip
    return live, aligned, live_length


def adapt_reference_to_calibration(target, calibration_row, torso_length, frame_shape):
    """Adjust reference limb lengths to the proportions captured in T-pose."""
    calibration = np.asarray(calibration_row, dtype=float).reshape(pose.NUM_LANDMARKS, 3)[:, :2].copy()
    frame_height, frame_width = frame_shape[:2]
    calibration *= np.array([frame_width, frame_height], dtype=float)
    shoulder_center = calibration[[11, 12]].mean(axis=0)
    hip_center = calibration[[23, 24]].mean(axis=0)
    calibration_torso = float(np.linalg.norm(shoulder_center - hip_center))
    if calibration_torso < 1e-6:
        return target

    segments = ((11, 13), (13, 15), (12, 14), (14, 16),
                (23, 25), (25, 27), (24, 26), (26, 28))
    for parent, child in segments:
        vector = target[child] - target[parent]
        current_length = float(np.linalg.norm(vector))
        calibrated_length = float(np.linalg.norm(calibration[child] - calibration[parent]))
        desired_length = calibrated_length / calibration_torso * torso_length
        if current_length > 1e-6 and desired_length > 1e-6:
            target[child] = target[parent] + vector / current_length * desired_length
    return target


def make_coach_panel(frame_shape, display_name: str, reference_row, live_row,
                     calibration_row, width: int):
    """Show the live pose against a torso-aligned target and mark deviations."""
    import cv2

    height, _ = frame_shape[:2]
    panel = np.full((height, width, 3), (28, 32, 38), dtype=np.uint8)
    cv2.putText(panel, "POSE COACH", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (100, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, display_name[:34], (16, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (235, 235, 235), 1, cv2.LINE_AA)

    if reference_row is None:
        cv2.putText(panel, "No reference pose available", (16, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (180, 180, 180), 1, cv2.LINE_AA)
        return panel

    if live_row is None:
        cv2.putText(panel, "Move into camera view", (16, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (180, 180, 180), 1, cv2.LINE_AA)
        return panel

    aligned_result = align_reference_to_person(reference_row, live_row, frame_shape)
    if aligned_result is None:
        cv2.putText(panel, "Pose could not be aligned", (16, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (180, 180, 180), 1, cv2.LINE_AA)
        return panel
    live, target, torso_length = aligned_result
    target = adapt_reference_to_calibration(target, calibration_row, torso_length, frame_shape)
    target, live = level_target_and_live(target, live)
    errors = np.linalg.norm(live - target, axis=1)
    threshold = torso_length * 0.18
    moving_indices = set(np.flatnonzero(errors > threshold).tolist())
    joints = sorted({point for edge in POSE_CONNECTIONS for point in edge})

    points_for_bounds = np.vstack((live[joints], target[joints]))
    low = points_for_bounds.min(axis=0)
    high = points_for_bounds.max(axis=0)
    span = np.maximum(high - low, 1e-4)
    left, top, right, bottom = 24, 90, width - 24, height - 82
    draw_width, draw_height = right - left, bottom - top
    scale = min(draw_width / span[0], draw_height / span[1])
    offset_x = left + (draw_width - span[0] * scale) / 2
    offset_y = top + (draw_height - span[1] * scale) / 2

    def to_panel(points):
        return [
            (int(offset_x + (point[0] - low[0]) * scale),
             int(offset_y + (point[1] - low[1]) * scale))
            for point in points
        ]

    live_points = to_panel(live)
    target_points = to_panel(target)
    for start, end in POSE_CONNECTIONS:
        cv2.line(panel, live_points[start], live_points[end], (125, 145, 165), 2, cv2.LINE_AA)
        needs_adjustment = start in moving_indices or end in moving_indices
        segment_color = (40, 40, 255) if needs_adjustment else (95, 205, 150)
        cv2.line(panel, target_points[start], target_points[end], segment_color, 3, cv2.LINE_AA)

    for index in joints:
        if index in moving_indices:
            cv2.line(panel, live_points[index], target_points[index], (40, 40, 255), 1, cv2.LINE_AA)
        cv2.circle(panel, live_points[index], 3, (160, 175, 190), -1, cv2.LINE_AA)
        target_color = (40, 40, 255) if index in moving_indices else (100, 230, 170)
        cv2.circle(panel, target_points[index], 5, target_color, -1, cv2.LINE_AA)

    cv2.putText(panel, "Red marks joints farther from the target", (16, height - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (205, 210, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "Target follows your torso position", (16, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 210, 190), 1, cv2.LINE_AA)
    return panel


def open_serial(port: str, baud: int):
    import serial

    print(f"Connecting to Arduino on {port}...")
    connection = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # opening the port reboots the Arduino; let it come up
    print("Arduino connected.")
    return connection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="models/exercise_models.pkl", help="Trained model bundle from train.py")
    parser.add_argument("--features", default="models/feature_columns.json", help="Feature column list from train.py")
    parser.add_argument("--data-dir", default="data", help="Per-exercise CSVs used for safe reference poses")
    parser.add_argument("--exercise-names", default="data/exercise_names.json",
                        help="Editable display-name mapping; inferred names are marked as such")
    parser.add_argument("--task-model", default=str(pose.DEFAULT_TASK_PATH), help="PoseLandmarker .task model path")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    parser.add_argument("--port", default="COM7", help="Arduino serial port (default COM7)")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum prediction confidence to actuate hardware (default 0.6)")
    parser.add_argument("--smooth-window", type=int, default=7,
                        help="Recent frames averaged for stable predictions (default 7)")
    parser.add_argument("--confirm-frames", type=int, default=3,
                        help="Frames required before a label change is accepted (default 3)")
    parser.add_argument("--no-serial", action="store_true", help="Run without an Arduino connected")
    args = parser.parse_args()

    import cv2

    master_payload = joblib.load(args.models)
    exercises = list(master_payload.keys())
    if not exercises:
        raise SystemExit(f"No exercises found in {args.models}. Run `python -m formcoach.train` first.")

    with open(args.features) as f:
        feature_columns = json.load(f)
    expected_len = len(feature_columns)

    current_exercise = exercises[0]
    reference_poses = {
        name: load_reference_poses(args.data_dir, name, feature_columns)
        for name in exercises
    }
    exercise_names = load_exercise_names(args.exercise_names)
    reference_width = 600
    coach_enabled = False
    calibrating = False
    calibration_row = None
    calibrator = TposeCalibrator()
    print("Exercises:")
    for i, name in enumerate(exercises, start=1):
        print(f"  {i}: {exercise_names.get(name, name)} [{name}]")
    print(f"Press 1-{len(exercises)} to switch, t to toggle the pose coach, q to quit.")

    arduino = None
    if not args.no_serial:
        try:
            arduino = open_serial(args.port, args.baud)
        except Exception as e:  # noqa: BLE001 - any serial failure degrades to software-only
            print(f"Serial connection failed ({e}). Continuing without hardware output.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try a different --camera index.")

    last_sent = None
    smoother = PredictionSmoother(args.smooth_window, args.confirm_frames)
    with pose.create_landmarker(args.task_model) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = pose.detect(landmarker, frame, int(time.time() * 1000))
            prediction, confidence = "No pose detected", 0.0

            row = pose.result_to_row(result)
            was_calibrating = calibrating
            calibration_valid = False
            calibration_progress = 0
            if calibrating:
                calibration_valid, calibration_progress, completed_calibration = calibrator.update(
                    result, row, frame.shape
                )
                if completed_calibration is not None:
                    calibration_row = completed_calibration
                    calibrating = False
                    coach_enabled = True
                    smoother.reset()
                    print("T-pose calibration complete. Pose coach enabled.")

            if was_calibrating:
                prediction = "Hold T-pose" if calibration_valid else "T-pose needed"
                if calibration_progress >= calibrator.REQUIRED_FRAMES:
                    prediction = "Calibration complete"
            if row is not None and len(row) == expected_len and not was_calibrating:
                model = master_payload[current_exercise]["model"]
                model_row = pose.canonicalize_row(row)
                probs = model.predict_proba(np.array(model_row).reshape(1, -1))[0]
                prediction, confidence = smoother.update(model.classes_, probs)

                # Actuate only on confident, changed predictions to avoid
                # flooding the serial link and chattering the hardware.
                if arduino is not None and confidence >= args.threshold and prediction != last_sent:
                    try:
                        arduino.write(f"{prediction.lower()}\n".encode("utf-8"))
                        last_sent = prediction
                        print(f"-> Arduino: {prediction.lower()} ({confidence:.2f})")
                    except Exception as e:  # noqa: BLE001
                        print(f"Serial write failed: {e}")

                is_optimal = (prediction.casefold() == "optimal"
                              and confidence >= args.threshold)
                mannequin_color = (95, 205, 150) if is_optimal else (95, 205, 245)
                pose.draw_landmarks(frame, result, color=mannequin_color)
            elif row is None:
                smoother.reset()
            exercise_display = exercise_names.get(current_exercise, current_exercise)
            cv2.rectangle(frame, (0, 0), (550, 110), (0, 0, 0), -1)
            cv2.putText(frame, f"EXERCISE: {exercise_display}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Form: {prediction}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "1-9: exercise | t: pose coach | q: quit", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            if calibrating:
                reference_panel = make_calibration_panel(
                    frame.shape,
                    exercise_display,
                    reference_width,
                    calibration_progress,
                    calibrator.REQUIRED_FRAMES,
                    calibration_valid,
                )
            elif coach_enabled:
                coach_row = row if row is not None and len(row) == expected_len else None
                reference_panel = make_coach_panel(
                    frame.shape,
                    exercise_display,
                    reference_poses[current_exercise].get("Optimal"),
                    coach_row,
                    calibration_row,
                    reference_width,
                )
            else:
                reference_panel = make_reference_panel(
                    frame.shape,
                    f"{current_exercise}: {exercise_display}",
                    reference_poses[current_exercise],
                    reference_width,
                )
            display = cv2.hconcat([frame, reference_panel])
            cv2.imshow("formcoach - live form feedback", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                if calibrating:
                    calibrating = False
                    calibrator.reset()
                    print("T-pose calibration cancelled.")
                elif coach_enabled:
                    coach_enabled = False
                    calibration_row = None
                    print("Pose coach disabled.")
                else:
                    coach_enabled = False
                    calibrating = True
                    calibration_row = None
                    calibrator.reset()
                    smoother.reset()
                    print("T-pose calibration started. Stand upright with arms straight out.")
            if ord("1") <= key <= ord("9"):
                index = key - ord("1")
                if index < len(exercises):
                    current_exercise = exercises[index]
                    last_sent = None  # re-send state for the new exercise context
                    smoother.reset()
                    print(f"Switched to: {exercise_names.get(current_exercise, current_exercise)}")

    cap.release()
    cv2.destroyAllWindows()
    if arduino is not None:
        arduino.close()
        print("Serial connection closed.")


if __name__ == "__main__":
    main()
