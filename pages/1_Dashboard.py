# 1_Dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Crime Dashboard", layout="wide")

st.title("📊 Crime Data Dashboard")
st.write("This dashboard provides an overview of crime statistics.\nExplore totals, distributions, and trends.")

# -----------------------------
# Load dataset
# -----------------------------
dataset_path = "women-crimedataset-India.csv"

if not os.path.exists(dataset_path):
    st.error("❌ Dataset not found. Please add 'women-crimedataset-India.csv' in your project folder.")
    st.stop()

data = pd.read_csv(dataset_path)

# -----------------------------
# Define crime columns
# -----------------------------
crime_columns = [
    'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
    'KIDNAPPING & ABDUCTION', 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
    'KIDNAPPING AND ABDUCTION OF OTHERS', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES'
]

# Convert all crime columns to numeric (coerce errors)
for col in crime_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# -----------------------------
# Total crimes column
# -----------------------------
data['TOTAL_CRIMES'] = data[crime_columns].sum(axis=1)

# -----------------------------
# Filters
# -----------------------------
states = ['All'] + sorted(data['STATE/UT'].dropna().unique().tolist())
selected_state = st.selectbox("Select State/UT", states)

if selected_state != 'All':
    filtered_data = data[data['STATE/UT'] == selected_state]
else:
    filtered_data = data.copy()

# -----------------------------
# Dataset preview
# -----------------------------
st.subheader("🔎 Dataset Preview")
st.dataframe(filtered_data.head())

# -----------------------------
# Summary statistics
# -----------------------------
st.subheader("📈 Summary Statistics")
st.write(filtered_data[crime_columns].describe())

# -----------------------------
# Total Crimes by Type
# -----------------------------
st.subheader("🚨 Total Crimes by Type")
total_by_type = filtered_data[crime_columns].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=total_by_type.values, y=total_by_type.index, palette="viridis", ax=ax)
ax.set_xlabel("Total Cases")
ax.set_ylabel("Crime Type")
ax.set_title("Total Crimes by Type")
st.pyplot(fig)

# -----------------------------
# Yearly Crime Trends
# -----------------------------
st.subheader("📅 Yearly Crime Trends")
yearly_trends = filtered_data.groupby('YEAR')[crime_columns].sum().reset_index()
st.line_chart(yearly_trends.set_index('YEAR'))

# -----------------------------
# Insights
# -----------------------------
st.subheader("💡 Insights")
if not filtered_data.empty:
    most_reported = filtered_data[crime_columns].sum().idxmax()
    least_reported = filtered_data[crime_columns].sum().idxmin()
    st.write(f"🔝 Most reported crime: {most_reported}")
    st.write(f"🔻 Least reported crime: {least_reported}")
else:
    st.write("No data available for the selected state.")
