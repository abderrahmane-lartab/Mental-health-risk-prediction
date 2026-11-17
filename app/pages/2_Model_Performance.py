import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.components.visualizations import plot_confusion_matrix
from src.models import train_and_evaluate

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance & Comparison")
st.write("Compare our three models: **Logistic Regression**, **Random Forest**, and **XGBoost**.")

EVAL_PATH = PROJECT_ROOT / "reports" / "metrics" / "evaluation_metrics.json"
CV_PATH = PROJECT_ROOT / "reports" / "metrics" / "cv_scores.json"
PARAMS_PATH = PROJECT_ROOT / "reports" / "metrics" / "best_params.json"


# Load saved evaluation metrics, or train baseline models if the pipeline hasn't run yet
@st.cache_resource
def get_performance_data():
    if EVAL_PATH.exists():
        with open(EVAL_PATH, "r") as f:
            return json.load(f), True

    # Fallback: run baseline training so the page still works right away
    raw_results = train_and_evaluate()
    models_data = {}
    for name, res in raw_results.items():
        models_data[name] = {
            "accuracy": round(res["accuracy"], 4),
            "confusion_matrix": res["confusion_matrix"].tolist(),
            "classification_report": res["classification_report"],
        }
    best_model = max(models_data, key=lambda m: models_data[m]["accuracy"])
    return {"best_model": best_model, "models": models_data}, False


eval_data, is_tuned = get_performance_data()
best_model = eval_data["best_model"]
models_dict = eval_data["models"]

if is_tuned:
    st.success(f"🏆 Best Tuned Model: **{best_model.replace('_', ' ').title()}**")
else:
    st.info(
        f"Showing baseline results (Best: **{best_model.replace('_', ' ').title()}**). "
        "Run the Airflow pipeline to generate tuned metrics."
    )

# Build a clean comparison table
st.subheader("1. Model Accuracy Comparison")
rows = []
for name, info in models_dict.items():
    row = {
        "Model": name.replace("_", " ").title(),
        "Accuracy": info["accuracy"],
    }
    if "precision" in info:
        row["Precision"] = info["precision"]
        row["Recall"] = info["recall"]
        row["F1 Score"] = info["f1_score"]
    rows.append(row)

comparison_df = pd.DataFrame(rows)
st.dataframe(comparison_df, use_container_width=True)
st.bar_chart(comparison_df.set_index("Model")["Accuracy"])

# Show Cross-Validation and Best Hyperparameters if available
if CV_PATH.exists() or PARAMS_PATH.exists():
    st.divider()
    col1, col2 = st.columns(2)

    if CV_PATH.exists():
        with col1:
            st.subheader("2. Cross-Validation Scores (5-Fold)")
            with open(CV_PATH, "r") as f:
                cv_data = json.load(f)
            cv_rows = [
                {
                    "Model": k.replace("_", " ").title(),
                    "Mean Accuracy": v["mean_accuracy"],
                    "Std": v["std_accuracy"],
                }
                for k, v in cv_data.items()
            ]
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)

    if PARAMS_PATH.exists():
        with col2:
            st.subheader("3. Best Hyperparameters (GridSearchCV)")
            with open(PARAMS_PATH, "r") as f:
                params_data = json.load(f)
            st.json(params_data)

st.divider()

# Inspect a specific model's Confusion Matrix and Classification Report
st.subheader("4. Detailed Model Report")
selected_model = st.selectbox("Select a model:", list(models_dict.keys()))
model_info = models_dict[selected_model]

col_cm, col_rep = st.columns(2)

with col_cm:
    labels = ["High", "Low", "Medium"]
    fig_cm = plot_confusion_matrix(model_info["confusion_matrix"], labels)
    st.pyplot(fig_cm)

with col_rep:
    st.text("Classification Report:")
    st.code(model_info["classification_report"])
