# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------- Step 1: Load the dataset ----------------
data = pd.read_csv("women-crimedataset-India.csv")

# ---------------- Step 2: Define the crime columns ----------------
crime_columns = [
    'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
    'KIDNAPPING & ABDUCTION', 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
    'KIDNAPPING AND ABDUCTION OF OTHERS', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES'
]

# ---------------- Step 3: Convert crime columns to numeric ----------------
for col in crime_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert non-numeric to NaN

# Replace NaN with 0
data[crime_columns] = data[crime_columns].fillna(0)

# ---------------- Step 4: Create TOTAL_CRIMES ----------------
data['TOTAL_CRIMES'] = data[crime_columns].sum(axis=1)

# ---------------- Step 5: Encode target variable ----------------
# Example: create 'Crime_Level' based on TOTAL_CRIMES
def crime_level(total):
    if total <= 50:
        return "Low"
    elif total <= 200:
        return "Medium"
    else:
        return "High"

data['Crime_Level'] = data['TOTAL_CRIMES'].apply(crime_level)

# ---------------- Step 6: Prepare features and target ----------------
X = data[crime_columns]  # Only numeric crime features
y = data['Crime_Level']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------- Step 7: Split dataset ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ---------------- Step 8: Train RandomForestClassifier ----------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- Step 9: Save the trained model and label encoder ----------------
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model trained and saved successfully!")
