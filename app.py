# app.py
import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# 1. Load Model and Encoders
# -----------------------------
with open("crime_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("state_encoder.pkl", "rb") as f:
    le_state = pickle.load(f)

with open("district_encoder.pkl", "rb") as f:
    le_district = pickle.load(f)

# -----------------------------
# 2. Define Crime Columns
# -----------------------------
crime_columns = [
    'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
    'KIDNAPPING & ABDUCTION', 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
    'KIDNAPPING AND ABDUCTION OF OTHERS', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES'
]

# -----------------------------
# 3. App Layout
# -----------------------------
st.title("Crime Level Prediction in India")
st.write("Enter details below to predict the overall crime level (High / Medium / Low)")

# Sample list of states and districts from your encoder
state_list = list(le_state.classes_)
district_list = list(le_district.classes_)

selected_state = st.selectbox("Select State", state_list)
selected_district = st.selectbox("Select District", district_list)

# Numeric inputs for crime features
st.write("Enter crime counts for the following categories:")
crime_inputs = {}
for col in crime_columns:
    crime_inputs[col] = st.number_input(col, min_value=0, value=0, step=1)

# -----------------------------
# 4. Prediction
# -----------------------------
if st.button("Predict Crime Level"):
    # Encode state and district
    state_encoded = le_state.transform([selected_state])[0]
    district_encoded = le_district.transform([selected_district])[0]

    # Prepare feature dataframe
    input_data = pd.DataFrame([[state_encoded, district_encoded] + list(crime_inputs.values())],
                              columns=['STATE', 'DISTRICT'] + crime_columns)

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"The predicted overall crime level is: **{prediction}**")
