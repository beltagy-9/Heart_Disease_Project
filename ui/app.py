import streamlit as st
import joblib
import numpy as np

# تحميل الموديل
model = joblib.load("final_model.pkl")

st.title("💓 Heart Disease Prediction App")

# --- إدخالات المستخدم بنفس ترتيب final_features ---
id_val = st.number_input("Patient ID", min_value=1, value=1)

thalch = st.number_input("Max Heart Rate (thalch)", min_value=70, max_value=220, value=150)
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)

exang = st.selectbox("Exercise Induced Angina (exang)", ["No", "Yes"])
exang_val = 1 if exang == "Yes" else 0

ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)

dataset = st.selectbox("Dataset Source", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])
dataset_Hungary = 1 if dataset == "Hungary" else 0
dataset_Switzerland = 1 if dataset == "Switzerland" else 0
dataset_VA = 1 if dataset == "VA Long Beach" else 0

cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina"])
cp_atypical = 1 if cp == "atypical angina" else 0

sex = st.selectbox("Sex", ["Female", "Male"])
sex_male = 1 if sex == "Male" else 0

# --- بناء features بنفس ترتيب final_features ---
features = np.array([[id_val, thalch, oldpeak, exang_val, ca,
                      dataset_Hungary, cp_atypical, sex_male,
                      dataset_Switzerland, dataset_VA]])

# --- التنبؤ ---
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ Likely Heart Disease")
    else:
        st.success("✅ No Heart Disease Detected")
