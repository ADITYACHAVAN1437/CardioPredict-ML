import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("Scaler.pkl")
excepted_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction by BORUTO")
st.markdown("Give The Following Details")
age = st.slider("Age: ", 18, 100, 40)
sex = st.selectbox("SEX", ["M", "F"])  # noqa: RUF016
chest_pain = st.selectbox("Chest Pain Type: ", ["ATA", "NAP", "TA", "ASY"])
restingBP = st.number_input("Resting BP(mm Hg): ", 80, 200, 120)
cholestrol = st.number_input("Cholestrol(mg/dL): ", 100, 600, 200)
FastingBS = st.selectbox("Fasting Blood Sugar>120 md/dL: ", [0, 1])
restingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_HR = st.slider("Max Heart Rate", 60, 220, 150)
Exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak(ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": restingBP,
        "Cholesterol": cholestrol,
        "FastingBS": FastingBS,
        "MaxHR": max_HR,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + restingECG: 1,
        "ExerciseAngina_" + Exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }
    input_df = pd.DataFrame([raw_input])
    for col in excepted_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[excepted_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
