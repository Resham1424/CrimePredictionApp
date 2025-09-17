# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("women-crimedataset-India.csv")
print("Columns in dataset:", data.columns)

# -----------------------------
# 2. Encode Categorical Columns
# -----------------------------
le_state = LabelEncoder()
le_district = LabelEncoder()

data['STATE'] = le_state.fit_transform(data['STATE/UT'])
data['DISTRICT'] = le_district.fit_transform(data['DISTRICT'])

# -----------------------------
# 3. Convert Crime Columns to Numeric
# -----------------------------
crime_columns = [
    'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 'CUSTODIAL RAPE', 'OTHER RAPE',
    'KIDNAPPING & ABDUCTION', 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
    'KIDNAPPING AND ABDUCTION OF OTHERS', 'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES'
]

for col in crime_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# -----------------------------
# 4. Create Overall Crime Level
# -----------------------------
data['TOTAL_CRIME'] = data[crime_columns].sum(axis=1)

def crime_level(total):
    if total > 1000:
        return 'High'
    elif total > 300:
        return 'Medium'
    else:
        return 'Low'

data['CRIME_LEVEL'] = data['TOTAL_CRIME'].apply(crime_level)

# -----------------------------
# 5. Features and Target
# -----------------------------
feature_columns = ['STATE', 'DISTRICT'] + crime_columns
X = data[feature_columns]
y = data['CRIME_LEVEL']

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 8. Save Model and Encoders
# -----------------------------
with open("crime_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("state_encoder.pkl", "wb") as f:
    pickle.dump(le_state, f)

with open("district_encoder.pkl", "wb") as f:
    pickle.dump(le_district, f)

print("Model and encoders saved successfully!")
