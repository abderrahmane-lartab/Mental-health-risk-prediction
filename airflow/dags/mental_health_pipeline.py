from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add project root to Python path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append("/opt/airflow")

from src.models import (
    cross_validate_models,
    evaluate_models,
    save_best_model,
    train_and_evaluate,
)
from src.optimization import optimize_models
from src.preprocessing import ingest_data, preprocess_data


# Task 1: Load and validate the dataset
def task_ingest_data():
    ingest_data()
    return "Data ingestion completed"


# Task 2: Clean, encode, scale data and save artifacts
def task_preprocess_data():
    preprocess_data()
    return "Data preprocessing completed"


# Task 3: Train baseline models (Logistic Regression, Random Forest, XGBoost)
def task_train_models():
    train_and_evaluate()
    return "Model training completed"


# Task 4: Run 5-fold Stratified Cross Validation
def task_cross_validate():
    scores = cross_validate_models()
    return scores


# Task 5: Tune hyperparameters with GridSearchCV
def task_tune_hyperparameters():
    optimize_models()
    return "Hyperparameter tuning completed"


# Task 6: Evaluate tuned models on test data
def task_evaluate_models():
    results = evaluate_models()
    return f"Best model: {results['best_model']}"


# Task 7: Save the best model and final metrics
def task_save_model():
    summary = save_best_model()
    return f"Saved best model to {summary['best_model_path']}"


default_args = {
    "owner": "abdo",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="student_mental_health_pipeline",
    default_args=default_args,
    description="Student Mental Health ML Pipeline",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=task_ingest_data,
    )

    data_preprocessing = PythonOperator(
        task_id="data_preprocessing",
        python_callable=task_preprocess_data,
    )

    model_training = PythonOperator(
        task_id="model_training",
        python_callable=task_train_models,
    )

    cross_validation = PythonOperator(
        task_id="cross_validation",
        python_callable=task_cross_validate,
    )

    hyperparameter_tuning = PythonOperator(
        task_id="hyperparameter_tuning",
        python_callable=task_tune_hyperparameters,
    )

    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=task_evaluate_models,
    )

    model_saving = PythonOperator(
        task_id="model_saving",
        python_callable=task_save_model,
    )

    # Pipeline order
    (
        data_ingestion
        >> data_preprocessing
        >> model_training
        >> cross_validation
        >> hyperparameter_tuning
        >> model_evaluation
        >> model_saving
    )
