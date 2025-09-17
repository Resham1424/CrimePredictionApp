import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Crime Prediction", layout="wide")
st.title("🔮 Crime Level Prediction")

# -------------------------
# Load model & encoders
# -------------------------
try:
    with open("crime_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("state_encoder.pkl", "rb") as f:
        le_state = pickle.load(f)
    with open("district_encoder.pkl", "rb") as f:
        le_district = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model or encoders not found. Add .pkl files in project folder.")
    st.stop()

# -------------------------
# User input
# -------------------------
st.subheader("Enter Details for Prediction")
state_input = st.selectbox("State/UT", le_state.classes_)
district_input = st.selectbox("District", le_district.classes_)
year_input = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
murder = st.number_input("Murder cases", min_value=0, value=0)
attempt_murder = st.number_input("Attempt to Murder cases", min_value=0, value=0)
rape = st.number_input("Rape cases", min_value=0, value=0)
other_crimes = st.number_input("Other crimes", min_value=0, value=0)

# Predict button
if st.button("Predict Crime Level"):
    try:
        state_encoded = le_state.transform([state_input])[0]
        district_encoded = le_district.transform([district_input])[0]
        input_df = pd.DataFrame([[state_encoded, district_encoded, murder, attempt_murder, rape, other_crimes]],
                                columns=['STATE', 'DISTRICT', 'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'OTHER_CRIMES'])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Crime Level: **{prediction}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
