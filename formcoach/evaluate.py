"""Measure per-exercise frame-level accuracy without changing project artifacts.

This reproduces train.py's stratified hold-out procedure, train-only
augmentation, and StandardScaler + RandomForest pipeline. The untouched test
rows come from the same CSV recording pool, so this does not measure
generalization to new people or recording sessions.

Usage:
    python -m formcoach.evaluate
    python -m formcoach.evaluate --data-dir data --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from formcoach import pose
from formcoach.augment import augment_dataframe


def evaluate(csv_path: Path, test_size: float, copies: int, noise: float, seed: int) -> float:
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"])
    X = pd.DataFrame(
        [pose.canonicalize_row(row) for row in X.to_numpy()],
        columns=X.columns,
    )
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    train_df = X_train.copy()
    train_df.insert(0, "label", y_train.values)
    train_aug = augment_dataframe(
        train_df, copies=copies, noise_level=noise, rng=np.random.default_rng(seed)
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
    )
    model.fit(train_aug.drop(columns=["label"]), train_aug["label"])
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    classes = model.classes_.tolist()

    print(f"\n=== {csv_path.stem} ===")
    print(f"Raw samples: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%} ({int((predictions == y_test.to_numpy()).sum())}/{len(y_test)} correct)")
    print("\nPer-class report:")
    print(classification_report(y_test, predictions, labels=classes, zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=classes)
    print("Confusion matrix (rows=true, columns=predicted):")
    print("           " + " ".join(f"{label:>12}" for label in classes))
    for label, row in zip(classes, cm):
        print(f"{label:>10} " + " ".join(f"{int(value):12}" for value in row))
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data", help="Directory containing per-exercise CSV files")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {args.data_dir}/")

    scores = [evaluate(path, args.test_size, args.copies, args.noise, args.seed) for path in files]
    print("\n=== Summary ===")
    for path, score in zip(files, scores):
        print(f"{path.stem}: {score:.2%}")
    print(f"Unweighted mean across exercises: {np.mean(scores):.2%}")
    print("\nNote: this is a frame-level random split from the same recordings, not a new-person/session test.")


if __name__ == "__main__":
    main()
