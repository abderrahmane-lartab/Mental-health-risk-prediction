from pathlib import Path
import pandas as pd
import streamlit as st

# Set up page title and layout
st.set_page_config(
    page_title="Student Mental Health ML",
    page_icon="🧠",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"

st.title("🧠 Student Mental Health Risk Prediction")
st.write(
    "Welcome! This app explores university student lifestyle and academic data "
    "to predict their mental health risk level (**Low**, **Medium**, or **High**)."
)

st.divider()

# Load the raw dataset to show quick stats
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Features", df.shape[1] - 2)  # Excluding Student_ID and Target
    col3.metric("Target Classes", df["Mental_Health_Risk"].nunique())
    col4.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.warning("Dataset file not found in data/raw/student_mental_health.csv")

st.divider()

# Simple navigation guide
st.subheader("📌 How to use this app")
st.markdown(
    """
- **1. Data Analysis**: Explore the dataset, check missing values, and view charts.
- **2. Model Performance**: Compare Logistic Regression, Random Forest, and XGBoost results.
- **3. Prediction Demo**: Enter a student's information to predict their mental health risk.
"""
)
