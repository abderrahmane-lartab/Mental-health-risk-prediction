import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing_input.csv"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "preprocessor_artifacts.joblib"
TARGET_COLUMN = "Mental_Health_Risk"


def ingest_data(raw_path=RAW_DATA_PATH, output_path=DEFAULT_INPUT_PATH):
    """Load raw dataset, check columns and target variable, and save cleaned input."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found at {raw_path}")

    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    # Drop Student_ID column before preprocessing
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Data ingested: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(
    input_path=DEFAULT_INPUT_PATH,
    artifact_path=DEFAULT_ARTIFACT_PATH,
    verbose=True,
):
    """Split the input data, fit preprocessing pipelines, and save artifacts."""
    input_path = Path(input_path)
    if not input_path.exists():
        ingest_data(output_path=input_path)

    df = pd.read_csv(input_path)
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    num_cols = X_train.select_dtypes(include="number").columns
    ordinal_cols = ["Physical_Activity", "Diet_Quality"]
    ordinal_mapping = [
        ["Low", "Moderate", "High"],
        ["Poor", "Average", "Good"],
    ]
    nominal_cols = [
        "Gender",
        "Part_Time_Job",
        "Previous_Mental_Health_History",
    ]

    if verbose:
        logger.info(f"Train Data: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
        logger.info(f"Test Data: {X_test.shape[0]} rows, {X_test.shape[1]} columns")

    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=ordinal_mapping)),
            ("scaler", StandardScaler()),
        ]
    )

    nominal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numerical", numerical_pipeline, num_cols),
            ("ordinal", ordinal_pipeline, ordinal_cols),
            ("nominal", nominal_pipeline, nominal_cols),
        ]
    )

    preprocessor.fit_transform(X_train)
    preprocessor.transform(X_test)

    artifacts = {
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, artifact_path)

    if verbose:
        logger.info("Preprocessing artifacts saved successfully!")

    return artifacts


if __name__ == "__main__":
    ingest_data()
    preprocess_data()
