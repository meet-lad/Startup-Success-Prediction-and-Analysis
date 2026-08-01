"""Train and export the startup-success prediction pipeline.

Run from the repository root:
    python src/train_model.py

The exported pipeline includes preprocessing, so it accepts the five raw model
features used by the Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "processed" / "processed_dataset.csv"
MODEL_PATH = ROOT / "Models" / "startup_success_model.joblib"
METADATA_PATH = ROOT / "Models" / "startup_success_model_metadata.json"

NUMERIC_FEATURES = ["company_age", "Funding_log", "funding_rounds"]
CATEGORICAL_FEATURES = ["Industry_clean", "country_code"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the categorical cleaning used by the advanced-model notebook."""
    data = data.copy()
    data["country_code"] = data["country_code"].replace(r"^\s*$", np.nan, regex=True).fillna("Unknown")
    top_countries = data["country_code"].value_counts().head(5).index
    data["country_code"] = data["country_code"].where(data["country_code"].isin(top_countries), "Other")
    data["Industry_clean"] = data["Industry_clean"].replace(r"^\s*$", np.nan, regex=True).fillna("Other")
    return data


def train_and_export() -> tuple[Pipeline, dict]:
    """Train the final pipeline, persist it, and return it with metadata."""
    data = prepare_data(pd.read_csv(DATA_PATH))
    X = data[FEATURES]
    y = data["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=20,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 3),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    metadata = {
        "model": "RandomForestClassifier",
        "features": FEATURES,
        "top_countries": sorted(data.loc[data["country_code"] != "Other", "country_code"].unique().tolist()),
        "industries": sorted(data["Industry_clean"].unique().tolist()),
        "test_metrics": metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return pipeline, metadata


def main() -> None:
    train_and_export()
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print("Test metrics:", metadata["test_metrics"])


if __name__ == "__main__":
    main()
