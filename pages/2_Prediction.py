import streamlit as st
import pandas as pd
import pickle

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    with open("crime_prediction_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------- Define Features --------------------
feature_columns = [
    'MURDER',
    'ATTEMPT TO MURDER',
    'RAPE',
    'CUSTODIAL RAPE',
    'OTHER RAPE',
    'KIDNAPPING & ABDUCTION',
    'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
    'KIDNAPPING AND ABDUCTION OF OTHERS',
    'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN',
    'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES'
]

# -------------------- Streamlit UI --------------------
st.title("🔮 Crime Prediction App")
st.write("Enter crime statistics to predict crime level.")

# Collect user inputs
inputs = {}
for col in feature_columns:
    inputs[col] = st.number_input(f"{col}", min_value=0, value=0, step=1)

# Convert inputs into DataFrame (with same feature order)
input_df = pd.DataFrame([inputs], columns=feature_columns)

# -------------------- Prediction --------------------
if st.button("Predict Crime Level"):
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted Crime Level: **{prediction}**")
