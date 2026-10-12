import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for shadow and card layout styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained KNN model
@st.cache_resource
def load_model():
  with open("knn.pkl", "rb") as file:
    model = pickle.load(file)
  return model


model = load_model()

# Header Section
st.markdown(
    "<h1 style='text-align: center; color: #1f2937;'>🎓 Student Performance"
    " Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #4b5563;'>Provide student details"
    " below to estimate performance using your trained model.</p>",
    unsafe_allow_html=True,
)

# Input Form inside a styled card container
with st.container():
  st.markdown('<div class="card">', unsafe_allow_html=True)
  st.subheader("Student Parameters")

  col1, col2 = st.columns(2)

  with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    course = st.selectbox(
        "Course", ["Science", "Arts", "Commerce", "Engineering", "Medicine"]
    )
    study_hours = st.number_input(
        "Study Hours", min_value=0.0, max_value=24.0, value=4.0
    )
    class_attendance = st.slider(
        "Class Attendance (%)", min_value=0, max_value=100, value=85
    )
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])

  with col2:
    sleep_hours = st.number_input(
        "Sleep Hours", min_value=0.0, max_value=24.0, value=7.0
    )
    sleep_quality = st.selectbox(
        "Sleep Quality", ["Poor", "Average", "Good", "Excellent"]
    )
    study_method = st.selectbox(
        "Study Method", ["Self-Study", "Online", "Group", "Classes"]
    )
    facility_rating = st.slider(
        "Facility Rating", min_value=1, max_value=5, value=3
    )
    exam_difficulty = st.selectbox(
        "Exam Difficulty", ["Easy", "Medium", "Hard"]
    )

  st.markdown("</div>", unsafe_allow_html=True)

# Prediction Section
if st.button("Run Prediction"):
  st.markdown('<div class="card">', unsafe_allow_html=True)
  st.subheader("Results")

  st.success(
      "Model successfully loaded! Connect your feature encoding and preprocessing"
      " pipeline here to output the prediction."
  )

  st.markdown("</div>", unsafe_allow_html=True)
