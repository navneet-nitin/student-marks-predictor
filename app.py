import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Student Marks Prediction 📊")

# --- COLLECT INPUTS ---
st.subheader("Daily Schedule (Hours)")

# 1. Hour-based Inputs
study_hours = st.slider("Study Hours", 0, 12, 5)
social_media = st.slider("Social Media Hours", 0, 10, 4)
netflix = st.slider("Netflix Hours", 0.0, 10, 1)
sleep = st.slider("Sleep Hours", 3, 12, 8)

# --- 24-HOUR VALIDATION LOGIC ---
total_hours = study_hours + social_media + netflix + sleep

# Show a progress bar or metric for hours used
st.write(f"**Total Hours Used:** {total_hours:.1f} / 24.0")

if total_hours > 24:
    st.error(f"⚠️ Error: You have entered {total_hours} hours! A day only has 24 hours. Please reduce time in some activities.")
    st.stop()  # This stops the app here, preventing the Predict button from appearing
else:
    st.success("✅ Time schedule is valid.")

# --- OTHER INPUTS ---
st.subheader("Other Details")
age = st.number_input("Age", 15, 30, 20)
part_time = st.selectbox("Part Time Job", ["No", "Yes"])
attendance = st.slider("Attendance %", 50, 100, 85)
exercise = st.slider("Exercise Frequency", 0, 7, 3)
mental = st.slider("Mental Health Rating", 1, 10, 7)
extra = st.selectbox("Extracurricular", ["No", "Yes"])

diet = st.selectbox("Diet Quality", ["Poor", "Fair", "Good"])
parent_edu = st.selectbox("Parental Education", ["High School", "Bachelor", "Master"])
internet = st.selectbox("Internet Quality", ["Poor", "Average", "Good"])
gender = st.selectbox("Gender", ["Female", "Male", "Other"])

# --- MAPPING ---
diet_map = {"Poor": 0, "Fair": 1, "Good": 2}
parent_map = {"High School": 0, "Bachelor": 1, "Master": 2}
internet_map = {"Poor": 0, "Average": 1, "Good": 2}
gender_map = {"Female": 0, "Male": 1, "Other": 2}

# --- PREDICTION ---
if st.button("Predict Score"):
    input_dict = {
        "age": age,
        "study_hours_per_day": study_hours,
        "social_media_hours": social_media,
        "netflix_hours": netflix,
        "part_time_job": 1 if part_time == "Yes" else 0,
        "attendance_percentage": attendance,
        "sleep_hours": sleep,
        "exercise_frequency": exercise,
        "mental_health_rating": mental,
        "extracurricular_participation": 1 if extra == "Yes" else 0,
        "diet_quality_num": diet_map[diet],
        "parental_education_level_num": parent_map[parent_edu],
        "internet_quality_num": internet_map[internet],
        "gender_num": gender_map[gender]
    }

    input_df = pd.DataFrame([input_dict], columns=feature_names)
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    final_result = max(0, min(100, prediction[0]))
    
    # --- LOGIC FOR EFFECTS ---
    if final_result >= 80:
        st.balloons()  # 🎈 Party time for high scores!
        st.success(f"🎉 Amazing Job! Your predicted score is {final_result:.2f}%")
    else:
        st.snow()      # ❄️ Calm snow for lower scores (encouragement)

        st.info(f"❄️ Good effort! Your predicted score is {final_result:.2f}%")
