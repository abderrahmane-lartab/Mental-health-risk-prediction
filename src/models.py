import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "preprocessor_artifacts.joblib"
DEFAULT_BASELINE_MODELS_PATH = PROJECT_ROOT / "models" / "baseline_models.joblib"
DEFAULT_OPTIMIZED_MODELS_PATH = PROJECT_ROOT / "models" / "optimized_models.joblib"
DEFAULT_BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
CV_SCORES_PATH = METRICS_DIR / "cv_scores.json"
BEST_PARAMS_PATH = METRICS_DIR / "best_params.json"
EVALUATION_METRICS_PATH = METRICS_DIR / "evaluation_metrics.json"
EXPERIMENT_SUMMARY_PATH = METRICS_DIR / "experiment_summary.json"


def get_baseline_models():
    """Return the three baseline models used in the project."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42, C=0.01
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
    }


def train_and_evaluate(artifact_path=DEFAULT_ARTIFACT_PATH):
    """Train Logistic Regression, Random Forest, and XGBoost on the training data."""
    artifacts = joblib.load(artifact_path)
    preprocessor = artifacts["preprocessor"]
    X_train = artifacts["X_train"]
    X_test = artifacts["X_test"]
    y_train = artifacts["y_train"]
    y_test = artifacts["y_test"]

    models = get_baseline_models()

    # 1. Logistic Regression
    logistic_pipeline = Pipeline(
        [("preprocessor", clone(preprocessor)), ("model", models["logistic_regression"])]
    )
    logistic_pipeline.fit(X_train, y_train)
    y_pred = logistic_pipeline.predict(X_test)

    # 2. Random Forest
    rf_pipeline = Pipeline(
        [("preprocessor", clone(preprocessor)), ("model", models["random_forest"])]
    )
    rf_pipeline.fit(X_train, y_train)
    y_rf_pred = rf_pipeline.predict(X_test)

    # 3. XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    xgb_pipeline = Pipeline(
        [("preprocessor", clone(preprocessor)), ("model", models["xgboost"])]
    )
    xgb_pipeline.fit(X_train, y_train_encoded)
    y_xgb_pred = xgb_pipeline.predict(X_test)

    results = {
        "logistic_regression": {
            "pipeline": logistic_pipeline,
            "predictions": y_pred,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        },
        "random_forest": {
            "pipeline": rf_pipeline,
            "predictions": y_rf_pred,
            "accuracy": float(accuracy_score(y_test, y_rf_pred)),
            "classification_report": classification_report(y_test, y_rf_pred),
            "confusion_matrix": confusion_matrix(y_test, y_rf_pred),
        },
        "xgboost": {
            "pipeline": xgb_pipeline,
            "label_encoder": label_encoder,
            "predictions": y_xgb_pred,
            "accuracy": float(accuracy_score(y_test_encoded, y_xgb_pred)),
            "classification_report": classification_report(y_test_encoded, y_xgb_pred),
            "confusion_matrix": confusion_matrix(y_test_encoded, y_xgb_pred),
        },
    }

    for name, res in results.items():
        logger.info(f"{name} accuracy: {res['accuracy']:.4f}")

    DEFAULT_BASELINE_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": {k: v["pipeline"] for k, v in results.items()},
            "label_encoder": label_encoder,
        },
        DEFAULT_BASELINE_MODELS_PATH,
    )

    return results


def cross_validate_models(
    artifact_path=DEFAULT_ARTIFACT_PATH, cv_scores_path=CV_SCORES_PATH
):
    """Run 5-fold StratifiedKFold cross-validation and save the scores."""
    artifacts = joblib.load(artifact_path)
    preprocessor = artifacts["preprocessor"]
    X_train = artifacts["X_train"]
    y_train = artifacts["y_train"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_baseline_models()
    y_train_encoded = LabelEncoder().fit_transform(y_train)

    cv_scores = {}

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])
        target = y_train_encoded if name == "xgboost" else y_train

        scores = cross_val_score(pipeline, X_train, target, cv=cv, scoring="accuracy")
        cv_scores[name] = {
            "fold_scores": [round(float(s), 4) for s in scores],
            "mean_accuracy": round(float(np.mean(scores)), 4),
            "std_accuracy": round(float(np.std(scores)), 4),
        }
        logger.info(
            f"{name} CV mean accuracy: {cv_scores[name]['mean_accuracy']:.4f}"
        )

    cv_scores_path = Path(cv_scores_path)
    cv_scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_scores_path, "w") as f:
        json.dump(cv_scores, f, indent=2)

    return cv_scores


def evaluate_models(
    artifact_path=DEFAULT_ARTIFACT_PATH,
    models_path=DEFAULT_OPTIMIZED_MODELS_PATH,
    evaluation_path=EVALUATION_METRICS_PATH,
):
    """Evaluate tuned models on the test set and compare their metrics."""
    artifacts = joblib.load(artifact_path)
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    saved_data = joblib.load(models_path)
    models = saved_data["models"]
    label_encoder = saved_data["label_encoder"]

    evaluation_results = {}

    for name, pipeline in models.items():
        preds = pipeline.predict(X_test)
        if name == "xgboost":
            preds = label_encoder.inverse_transform(preds.astype(int))

        evaluation_results[name] = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(
                float(precision_score(y_test, preds, average="weighted")), 4
            ),
            "recall": round(float(recall_score(y_test, preds, average="weighted")), 4),
            "f1_score": round(float(f1_score(y_test, preds, average="weighted")), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds),
        }
        logger.info(
            f"{name} Test Accuracy: {evaluation_results[name]['accuracy']:.4f} | "
            f"F1: {evaluation_results[name]['f1_score']:.4f}"
        )

    # Pick the model with the highest test accuracy
    best_model = max(
        evaluation_results, key=lambda m: evaluation_results[m]["accuracy"]
    )

    output = {
        "best_model": best_model,
        "models": evaluation_results,
    }

    evaluation_path = Path(evaluation_path)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evaluation_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


def save_best_model(
    artifact_path=DEFAULT_ARTIFACT_PATH,
    optimized_models_path=DEFAULT_OPTIMIZED_MODELS_PATH,
    evaluation_path=EVALUATION_METRICS_PATH,
    best_model_path=DEFAULT_BEST_MODEL_PATH,
    summary_path=EXPERIMENT_SUMMARY_PATH,
):
    """Save the best-performing model and experiment summary."""
    artifacts = joblib.load(artifact_path)
    saved_data = joblib.load(optimized_models_path)

    with open(evaluation_path, "r") as f:
        eval_results = json.load(f)

    best_name = eval_results["best_model"]
    best_pipeline = saved_data["models"][best_name]

    best_model_path = Path(best_model_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_pipeline,
            "preprocessor": artifacts["preprocessor"],
            "label_encoder": saved_data["label_encoder"]
            if best_name == "xgboost"
            else None,
            "metrics": eval_results["models"][best_name],
        },
        best_model_path,
    )

    summary = {
        "best_model": best_name,
        "best_model_path": str(best_model_path),
        "metrics": eval_results["models"][best_name],
        "all_models": eval_results["models"],
    }

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Best model ({best_name}) saved to {best_model_path}")
    return summary
