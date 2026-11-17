from pathlib import Path
import sys
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.components.visualizations import (
    plot_correlation_heatmap,
    plot_feature_distribution,
    plot_target_distribution,
)

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.write("Here we explore the student dataset, missing values, and feature relationships.")

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

# Show summary statistics
st.subheader("1. Summary Statistics")
st.dataframe(df.describe().T, use_container_width=True)

# Show missing values and target balance side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Missing Values per Column")
    missing = df.isna().sum()
    missing = missing[missing > 0].reset_index()
    missing.columns = ["Column", "Missing Count"]
    st.dataframe(missing, use_container_width=True)

with col2:
    st.subheader("3. Target Class Distribution")
    fig_target = plot_target_distribution(df)
    st.pyplot(fig_target)

st.divider()

# Let the user pick a column to plot its distribution
st.subheader("4. Feature Distributions")
num_cols = df.select_dtypes(include="number").columns.tolist()
selected_col = st.selectbox("Choose a feature to visualize:", num_cols)

fig_dist = plot_feature_distribution(df, selected_col)
st.pyplot(fig_dist)

st.divider()

# Show correlation heatmap
st.subheader("5. Correlation Heatmap")
fig_corr = plot_correlation_heatmap(df)
st.pyplot(fig_corr)
