import streamlit as st
import pandas as pd

st.set_page_config(page_title="Crime Dashboard", layout="wide")
st.title("📊 Crime Data Dashboard")

# Load dataset
data = pd.read_csv("women-crimedataset-India.csv")

crime_columns = [
    'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
    'KIDNAPPING & ABDUCTION', 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
    'KIDNAPPING AND ABDUCTION OF OTHERS', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES'
]

# Total crimes per state
data['TOTAL_CRIMES'] = data[crime_columns].sum(axis=1)
state_crime_sum = data.groupby("STATE/UT")["TOTAL_CRIMES"].sum().sort_values(ascending=False).head(10)

st.subheader("Top 10 States with Highest Crimes")
st.bar_chart(state_crime_sum)

# Yearly trend
yearly_trend = data.groupby("YEAR")["TOTAL_CRIMES"].sum()
st.subheader("Yearly Trend of Crimes")
st.line_chart(yearly_trend)

# Crime type distribution
crime_type_sum = data[crime_columns].sum().sort_values(ascending=False)
st.subheader("Crime Type Distribution")
st.bar_chart(crime_type_sum)
