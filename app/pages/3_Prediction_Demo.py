from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models import train_and_evaluate

st.set_page_config(page_title="Prediction Demo", page_icon="🔮", layout="wide")
st.title("🔮 Student Mental Health Risk Predictor")
st.write("Fill in the student details below and click **Predict** to estimate their mental health risk.")

BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
BASELINE_MODELS_PATH = PROJECT_ROOT / "models" / "baseline_models.joblib"


# Load the saved best model, or train baseline Logistic Regression if not saved yet
@st.cache_resource
def load_prediction_model():
    if BEST_MODEL_PATH.exists():
        data = joblib.load(BEST_MODEL_PATH)
        return data["pipeline"], data["model_name"], data.get("label_encoder")

    if BASELINE_MODELS_PATH.exists():
        data = joblib.load(BASELINE_MODELS_PATH)
        return data["models"]["logistic_regression"], "logistic_regression", None

    results = train_and_evaluate()
    return results["logistic_regression"]["pipeline"], "logistic_regression", None


pipeline, model_name, label_encoder = load_prediction_model()
st.caption(f"Using model: **{model_name.replace('_', ' ').title()}**")

# Input form for student features
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal & Academic")
        age = st.slider("Age", 17, 30, 21)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        year_of_study = st.slider("Year of Study", 1, 5, 2)
        gpa = st.slider("GPA (4.0 scale)", 1.5, 4.0, 2.8, 0.05)
        attendance = st.slider("Attendance Rate (%)", 40.0, 100.0, 85.0, 0.5)

    with col2:
        st.subheader("💤 Lifestyle & Habits")
        sleep_hours = st.slider("Average Nightly Sleep (Hours)", 3.0, 10.0, 7.0, 0.5)
        study_hours = st.slider("Study Hours per Week", 2.0, 45.0, 20.0, 0.5)
        physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
        part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])

    with col3:
        st.subheader("📱 Social & Stress Factors")
        social_media = st.slider("Daily Social Media (Hours)", 0.5, 10.0, 5.0, 0.5)
        screen_time = st.slider("Total Daily Screen Time (Hours)", 1.0, 14.0, 8.0, 0.5)
        financial_stress = st.slider("Financial Stress (1-10)", 1.0, 10.0, 5.0, 0.5)
        family_support = st.slider("Family Support (1-10)", 1.0, 10.0, 7.0, 0.5)
        prev_history = st.selectbox("Previous Mental Health History", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Risk Level", use_container_width=True)

if submitted:
    # Create a single-row DataFrame matching the training columns
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Year_of_Study": year_of_study,
                "Sleep_Hours": sleep_hours,
                "Study_Hours_Per_Week": study_hours,
                "GPA": gpa,
                "Social_Media_Hours": social_media,
                "Screen_Time_Hours": screen_time,
                "Physical_Activity": physical_activity,
                "Financial_Stress": financial_stress,
                "Family_Support": family_support,
                "Attendance_Rate": attendance,
                "Part_Time_Job": part_time_job,
                "Diet_Quality": diet_quality,
                "Previous_Mental_Health_History": prev_history,
            }
        ]
    )

    prediction = pipeline.predict(input_df)[0]
    if label_encoder is not None:
        prediction = label_encoder.inverse_transform([int(prediction)])[0]

    st.divider()
    st.subheader("Prediction Result")

    # Show colored alert based on predicted risk
    if prediction == "Low":
        st.success("🟢 **Predicted Mental Health Risk: LOW**")
    elif prediction == "Medium":
        st.warning("🟡 **Predicted Mental Health Risk: MEDIUM**")
    else:
        st.error("🔴 **Predicted Mental Health Risk: HIGH**")

    # Show prediction probabilities if supported by the model
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        classes = (
            label_encoder.classes_
            if label_encoder is not None
            else pipeline.classes_
        )
        prob_df = pd.DataFrame({"Risk Level": classes, "Probability": probs})
        st.write("**Confidence Scores:**")
        st.bar_chart(prob_df.set_index("Risk Level"))
