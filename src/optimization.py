import json
import logging
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.models import (
    BEST_PARAMS_PATH,
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_OPTIMIZED_MODELS_PATH,
    PROJECT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def optimize_models(
    artifact_path=DEFAULT_ARTIFACT_PATH,
    optimized_models_path=DEFAULT_OPTIMIZED_MODELS_PATH,
    best_params_path=BEST_PARAMS_PATH,
):
    """Grid-search Logistic Regression, Random Forest, and XGBoost."""
    artifacts = joblib.load(artifact_path)
    preprocessor = artifacts["preprocessor"]
    X_train = artifacts["X_train"]
    y_train = artifacts["y_train"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    models_and_grids = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"model__C": [0.01, 0.1, 1, 10]},
            y_train,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            },
            y_train,
        ),
        "xgboost": (
            XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=-1),
            {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 4, 6],
            },
            y_train_encoded,
        ),
    }

    results = {}
    best_models = {}
    best_params = {}

    for name, (model, param_grid, target) in models_and_grids.items():
        pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            refit=True,
            n_jobs=-1,
        )
        search.fit(X_train, target)

        results[name] = search
        best_models[name] = search.best_estimator_
        best_params[name] = {
            "best_score": round(float(search.best_score_), 4),
            "best_params": search.best_params_,
        }

        logger.info(f"{name}: best CV accuracy = {search.best_score_:.4f}")
        logger.info(f"Best parameters: {search.best_params_}")

    # Save tuned models
    optimized_models_path = Path(optimized_models_path)
    optimized_models_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"models": best_models, "label_encoder": label_encoder},
        optimized_models_path,
    )

    # Save best parameters JSON
    best_params_path = Path(best_params_path)
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    return results


if __name__ == "__main__":
    optimize_models()
