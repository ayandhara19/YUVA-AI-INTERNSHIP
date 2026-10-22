"""
Week 2: Supervised Machine Learning Models
Dataset: UCI Student Performance (student-mat.csv)

Run from this folder:
    python supervised_ml.py

Expected input:
    ../Week 1/student-mat.csv
or set the DATA_PATH environment variable to the CSV path.

The CSV is semicolon-delimited. The script uses a train/test split and fits
preprocessing only on training data via scikit-learn Pipelines.
"""

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA_PATH = PROJECT_DIR.parent / "Week 1" / "student-mat.csv"
DATA_PATH = Path(os.environ.get("DATA_PATH", str(DEFAULT_DATA_PATH)))

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(path: Path) -> pd.DataFrame:
    """Load the original UCI mathematics student dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place student-mat.csv in the Week 1 folder, or set DATA_PATH "
            "to the full path of the CSV file."
        )
    df = pd.read_csv(path, sep=";")
    if "G3" not in df.columns:
        raise ValueError("The dataset must contain the final-grade column 'G3'.")
    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute missing values, scale numeric features, and encode categories."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def save_regression_plot(y_true, predictions, output_path: Path) -> None:
    """Save actual-vs-predicted scatter plot."""
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, predictions, alpha=0.7)
    low = min(float(np.min(y_true)), float(np.min(predictions)))
    high = max(float(np.max(y_true)), float(np.max(predictions)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual final grade (G3)")
    plt.ylabel("Predicted final grade (G3)")
    plt.title("Regression: Actual vs Predicted Final Grades")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate three regressors to predict G3."""
    # G1 and G2 are intentionally excluded to make prediction less dependent
    # on earlier grades, and to reduce the risk of an unrealistically easy task.
    excluded = ["G3", "G1", "G2"]
    X = df.drop(columns=excluded, errors="ignore")
    y = df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=RANDOM_STATE, max_depth=5
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, max_depth=8
        ),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    }

    rows = []
    prediction_table = pd.DataFrame({"actual_G3": y_test.reset_index(drop=True)})

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", make_preprocessor(X_train)), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
            }
        )
        prediction_table[name] = pred
        if name == "Random Forest Regressor":
            save_regression_plot(
                y_test.to_numpy(),
                pred,
                OUTPUT_DIR / "regression_actual_vs_predicted.png",
            )

    results = pd.DataFrame(rows).sort_values("RMSE")
    results.to_csv(OUTPUT_DIR / "regression_model_comparison.csv", index=False)
    prediction_table.to_csv(OUTPUT_DIR / "regression_predictions.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(results["model"], results["RMSE"])
    plt.ylabel("RMSE (lower is better)")
    plt.title("Regression Model Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regression_model_comparison.png", dpi=160)
    plt.close()

    return results


def run_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate four classifiers to predict pass/fail."""
    # Pass is defined as G3 >= 10. G3 must not be included in X because it
    # directly defines y; G1 and G2 are also excluded to avoid grade leakage.
    y = (df["G3"] >= 10).astype(int).rename("pass")
    X = df.drop(columns=["G3", "G1", "G2"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=5
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, max_depth=8
        ),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
    }

    rows = []
    predictions = {}
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", make_preprocessor(X_train)), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        predictions[name] = pred

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1_score": f1_score(y_test, pred, zero_division=0),
            }
        )

        if name == "Random Forest Classifier":
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            display = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Fail (<10)", "Pass (>=10)"]
            )
            display.plot()
            plt.title("Random Forest Classification Confusion Matrix")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "classification_confusion_matrix.png", dpi=160)
            plt.close()

            report = classification_report(
                y_test,
                pred,
                labels=[0, 1],
                target_names=["Fail (<10)", "Pass (>=10)"],
                zero_division=0,
            )
            (OUTPUT_DIR / "classification_report.txt").write_text(
                report, encoding="utf-8"
            )

    results = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    results.to_csv(OUTPUT_DIR / "classification_model_comparison.csv", index=False)

    prediction_table = pd.DataFrame(
        {
            "actual_pass": y_test.reset_index(drop=True),
            **{f"{name}_prediction": pd.Series(pred) for name, pred in predictions.items()},
        }
    )
    prediction_table.to_csv(OUTPUT_DIR / "classification_predictions.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(results["model"], results["f1_score"])
    plt.ylabel("F1-score (higher is better)")
    plt.title("Classification Model Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "classification_model_comparison.png", dpi=160)
    plt.close()

    return results


def main():
    print("=" * 65)
    print("WEEK 2: SUPERVISED MACHINE LEARNING")
    print("=" * 65)
    print(f"Loading dataset from: {DATA_PATH}")

    df = load_data(DATA_PATH)
    print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing cells: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("Pass threshold for classification: G3 >= 10")
    print("Train/test split: 80% / 20% (random_state=42)")

    regression_results = run_regression(df)
    classification_results = run_classification(df)

    print("\nREGRESSION MODEL COMPARISON")
    print(regression_results.round(3).to_string(index=False))
    print("\nCLASSIFICATION MODEL COMPARISON")
    print(classification_results.round(3).to_string(index=False))
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Review the CSV tables and graphs before writing final conclusions.")


if __name__ == "__main__":
    main()
