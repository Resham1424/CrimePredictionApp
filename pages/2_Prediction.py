import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Crime Prediction", layout="wide")
st.title("🤖 Crime Level Prediction")

# Load model & encoders
with open("crime_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("state_encoder.pkl", "rb") as f:
    le_state = pickle.load(f)

with open("district_encoder.pkl", "rb") as f:
    le_district = pickle.load(f)

# Load dataset for mapping
data = pd.read_csv("women-crimedataset-India.csv")
state_district_map = data.groupby('STATE/UT')['DISTRICT'].unique().to_dict()

# Crime columns
crime_columns = [
    'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
    'KIDNAPPING & ABDUCTION', 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
    'KIDNAPPING AND ABDUCTION OF OTHERS', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES'
]

# Sidebar inputs
st.sidebar.header("Input Parameters")
state_list = list(le_state.classes_)
selected_state = st.sidebar.selectbox("Select State", state_list)

district_list = state_district_map[selected_state]
selected_district = st.sidebar.selectbox("Select District", district_list, index=0)

crime_inputs = {}
for col in crime_columns:
    crime_inputs[col] = st.sidebar.number_input(col, min_value=0, value=50, step=1)

if st.sidebar.button("Predict Crime Level"):
    state_encoded = le_state.transform([selected_state])[0]
    district_encoded = le_district.transform([selected_district])[0]

    input_data = pd.DataFrame([[state_encoded, district_encoded] + list(crime_inputs.values())],
                              columns=['STATE', 'DISTRICT'] + crime_columns)

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.success(f"✅ The predicted crime level is: **{prediction}**")

    st.subheader("Crime Input Overview")
    crime_df = pd.DataFrame.from_dict(crime_inputs, orient='index', columns=['Count'])
    st.bar_chart(crime_df)
