import streamlit as st
import pandas as pd
import joblib

# Load trained model and label encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Crime input columns
crime_columns = [
    'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
    'KIDNAPPING & ABDUCTION', 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
    'KIDNAPPING AND ABDUCTION OF OTHERS', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES'
]

st.title("🔮 Crime Prediction App")

# Take inputs
inputs = {}
for col in crime_columns:
    inputs[col] = st.number_input(col, min_value=0, step=1)

# Predict button
if st.button("Predict Crime Level"):
    input_df = pd.DataFrame([inputs])
    prediction_encoded = model.predict(input_df)[0]
    prediction = le.inverse_transform([prediction_encoded])[0]
    st.success(f"✅ Predicted Crime Level: {prediction}")
