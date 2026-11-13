"""Pilot a subject-fold classifier on RehabPile's KIMORE arm-lifting task.

This is a separate binary, sequence-level experiment. It does not replace the
live five-class frame model. Data must first be downloaded into
data/rehabpile/KIMORE_clf_bn/LA using the official RehabPile archive.

Run with: python -m formcoach.rehabpile_experiment
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA_DIR = Path("data/rehabpile/KIMORE_clf_bn/LA")
OUTPUT_PATH = Path("assets/rehabpile_kimore_la_metrics.json")


def load_split(path: Path, split: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(path / f"x_{split}_fold{fold}.npy", allow_pickle=True)
    y = np.load(path / f"y_{split}_fold{fold}.npy", allow_pickle=True).reshape(-1)
    if x.ndim != 3 or x.shape[1] != 54:
        raise ValueError(f"Expected (samples, 54 joint channels, frames), got {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"Feature/label count mismatch in {path}, {split}")
    return np.asarray(x, dtype=np.float64), np.asarray(y)


def sequence_features(sequence: np.ndarray) -> np.ndarray:
    """Summarize a skeleton sequence after root-centering and scale normalization."""
    joints = sequence.reshape(18, 3, sequence.shape[-1]).transpose(2, 0, 1)
    relative = joints - joints[:, :1, :]
    body_scale = float(np.median(np.linalg.norm(relative, axis=2)))
    if not np.isfinite(body_scale) or body_scale < 1e-8:
        body_scale = 1.0
    relative /= body_scale

    velocity = np.diff(relative, axis=0)
    summaries = [
        np.mean(relative, axis=0),
        np.std(relative, axis=0),
        np.quantile(relative, 0.1, axis=0),
        np.quantile(relative, 0.9, axis=0),
        np.mean(np.abs(velocity), axis=0),
        np.std(velocity, axis=0),
    ]
    return np.concatenate([summary.reshape(-1) for summary in summaries])


def nearest_neighbor_predict(train_x, train_y, test_x, neighbors: int = 3):
    train_mean = train_x.mean(axis=0)
    train_scale = train_x.std(axis=0)
    train_scale[train_scale < 1e-8] = 1.0
    train_z = (train_x - train_mean) / train_scale
    test_z = (test_x - train_mean) / train_scale

    predictions = []
    labels, label_counts = np.unique(train_y, return_counts=True)
    class_weights = {
        label: np.sqrt(len(train_y) / (len(labels) * count))
        for label, count in zip(labels, label_counts)
    }
    for sample in test_z:
        distances = np.linalg.norm(train_z - sample, axis=1)
        nearest = np.argsort(distances)[:neighbors]
        votes = {label: 0.0 for label in labels}
        for index in nearest:
            label = train_y[index]
            votes[label] += class_weights[label] / max(float(distances[index]), 1e-6)
        predictions.append(max(votes, key=votes.get))
    return np.asarray(predictions), train_mean, train_scale, train_z


def metrics(y_true, y_pred, class_labels) -> dict:
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[class_labels.index(truth), class_labels.index(prediction)] += 1
    recalls = [
        matrix[i, i] / matrix[i].sum() if matrix[i].sum() else 0.0
        for i in range(len(class_labels))
    ]
    f1s = []
    for i in range(len(class_labels)):
        precision = matrix[i, i] / matrix[:, i].sum() if matrix[:, i].sum() else 0.0
        recall = recalls[i]
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "samples": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "classes": [int(label) for label in class_labels],
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
    }


def main() -> None:
    fold_results = []
    all_truth = []
    all_predictions = []
    all_majority_predictions = []
    full_dataset = None

    for fold in range(5):
        folder = DATA_DIR / f"fold{fold}"
        train_raw, train_y = load_split(folder, "train", fold)
        test_raw, test_y = load_split(folder, "test", fold)
        train_x = np.stack([sequence_features(sequence) for sequence in train_raw])
        test_x = np.stack([sequence_features(sequence) for sequence in test_raw])
        predictions, mean, scale, train_z = nearest_neighbor_predict(
            train_x, train_y, test_x, neighbors=3
        )
        labels = sorted(set(train_y.tolist()) | set(test_y.tolist()))
        result = metrics(test_y, predictions, labels)
        result["fold"] = fold
        result["train_samples"] = int(len(train_y))
        result["train_class_counts"] = {
            str(int(label)): int(np.sum(train_y == label)) for label in labels
        }
        fold_results.append(result)
        all_truth.extend(test_y.tolist())
        all_predictions.extend(predictions.tolist())
        majority = max(set(train_y.tolist()), key=train_y.tolist().count)
        all_majority_predictions.extend([majority] * len(test_y))
        if fold == 0:
            full_dataset = (
                np.vstack((train_x, test_x)),
                np.concatenate((train_y, test_y)),
            )
        print(f"fold {fold}: {result['samples']} test sequences, "
              f"accuracy={result['accuracy']:.3f}, "
              f"balanced_accuracy={result['balanced_accuracy']:.3f}, "
              f"macro_f1={result['macro_f1']:.3f}")

    class_labels = sorted(set(all_truth))
    pooled = metrics(np.asarray(all_truth), np.asarray(all_predictions), class_labels)
    majority_baseline = metrics(
        np.asarray(all_truth), np.asarray(all_majority_predictions), class_labels
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "RehabPile KIMORE_clf_bn_LA",
        "task": "binary sequence-level classification; experimental 3-nearest-neighbor baseline",
        "folds": fold_results,
        "pooled_subject_fold_metrics": pooled,
        "majority_baseline_same_folds": majority_baseline,
        "warning": "A separate pilot. Its binary sequence labels do not match the live 5-class frame labels.",
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if full_dataset is not None:
        full_x, full_y = full_dataset
        mean = full_x.mean(axis=0)
        scale = full_x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        full_z = (full_x - mean) / scale
        labels, counts = np.unique(full_y, return_counts=True)
        class_weights = np.asarray([
            np.sqrt(len(full_y) / (len(labels) * count)) for count in counts
        ])
        np.savez_compressed(
            Path("models/rehabpile_kimore_la_experimental.npz"),
            train_features=full_z,
            train_labels=full_y,
            class_labels=labels,
            class_weights=class_weights,
            feature_mean=mean,
            feature_scale=scale,
            neighbors=np.asarray([3]),
        )

    print(f"pooled held-out results: accuracy={pooled['accuracy']:.3f}, "
          f"balanced_accuracy={pooled['balanced_accuracy']:.3f}, "
          f"macro_f1={pooled['macro_f1']:.3f}")
    print(f"metrics -> {OUTPUT_PATH}")
    print("Experimental model saved separately; the live model was not overwritten.")


if __name__ == "__main__":
    main()
