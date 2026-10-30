"""Train frozen candidates and evaluate one validation-selected model once on test."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.schema import FEATURES, LABELS

ROOT = Path(__file__).resolve().parents[1]


def load_data(data_dir):
    """Join one-to-one records, removing identifiers from model features."""
    left = pd.read_excel(data_dir / "case_study1.xlsx")
    right = pd.read_excel(data_dir / "case_study2.xlsx")
    merged = left.merge(right, on="PROSPECTID", validate="one_to_one")
    if set(merged["Approved_Flag"].unique()) != set(LABELS):
        raise ValueError("Expected source labels P1, P2, P3, P4.")
    features = merged[list(FEATURES)].replace(-99999, np.nan)
    return features, merged["Approved_Flag"]


def metrics(model, features, labels):
    """Return multiclass performance with a fixed label order."""
    predicted = model.predict(features)
    return {
        "rows": len(labels),
        "accuracy": accuracy_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro"),
        "weighted_f1": f1_score(labels, predicted, average="weighted"),
        "classification_report": classification_report(
            labels, predicted, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=list(LABELS)
        ).tolist(),
    }


def train(data_dir, output_dir):
    """Fit preprocessing on training only; select using validation macro F1."""
    features, labels = load_data(data_dir)
    # Identical selected feature vectors stay in one partition, including missingness.
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    develop, test = next(split.split(features, labels, groups))
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    train_rel, valid_rel = next(
        split.split(features.iloc[develop], labels.iloc[develop], groups[develop])
    )
    training, validation = develop[train_rel], develop[valid_rel]
    partitions = {"train": training, "validation": validation, "test": test}
    for name, indexes in partitions.items():
        if set(labels.iloc[indexes]) != set(LABELS):
            raise ValueError(f"Missing class in {name} partition")
    assert set(groups[training]).isdisjoint(groups[test])
    assert set(groups[validation]).isdisjoint(groups[test])
    assert set(groups[training]).isdisjoint(groups[validation])
    candidates = {
        "majority_baseline": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            DummyClassifier(strategy="most_frequent"),
        ),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(
                n_estimators=180,
                max_depth=16,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                n_jobs=2,
                random_state=42,
            ),
        ),
    }
    comparison = {}
    for name, model in candidates.items():
        print("Fitting", name, flush=True)
        model.fit(features.iloc[training], labels.iloc[training])
        comparison[name] = metrics(
            model, features.iloc[validation], labels.iloc[validation]
        )
    winner = max(comparison, key=lambda name: comparison[name]["macro_f1"])
    model = candidates[winner]
    importance = permutation_importance(
        model,
        features.iloc[validation],
        labels.iloc[validation],
        scoring="f1_macro",
        n_repeats=3,
        random_state=42,
        n_jobs=2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "pipeline.joblib"
    joblib.dump(model, model_file, compress=3)
    source_hashes = {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ["case_study1.xlsx", "case_study2.xlsx"]
    }
    report = {
        "version": "1.0.0",
        "selected_model": winner,
        "seed": [42, 43],
        "source_sha256": source_hashes,
        "features": list(FEATURES),
        "classes": list(LABELS),
        "split": {
            name: {
                "rows": len(idx),
                "class_counts": labels.iloc[idx].value_counts().to_dict(),
            }
            for name, idx in partitions.items()
        },
        "split_strategy": (
            "60/20/20 approximately; identical feature rows grouped; seed 42/43"
        ),
        "validation_candidates": comparison,
        "test": metrics(model, features.iloc[test], labels.iloc[test]),
        "permutation_importance": dict(
            zip(FEATURES, importance.importances_mean.tolist())
        ),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "artifact_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    # Use medians for a demonstration input, not a real applicant record.
    sample = features.iloc[training].median().to_dict()
    sample = {
        field: value if field == "NETMONTHLYINCOME" else int(value)
        for field, value in sample.items()
    }
    (output_dir / "sample.json").write_text(json.dumps(sample, indent=2) + "\n")
    print(json.dumps({"winner": winner, "test": report["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset_and_materials")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    train(args.data_dir, args.output_dir)
