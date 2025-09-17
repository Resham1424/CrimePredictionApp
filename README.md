# Crime Prediction App

📊 **Overview**  
This project is a Machine Learning-powered web application that predicts crime levels in different districts of India. It also provides an interactive dashboard for visualizing crime statistics over the years.

---

## 🚀 Features

### 1. Crime Prediction
- Predicts crime level (**High**, **Medium**, **Low**) based on input features like:
  - State / UT
  - District
  - Year
  - Crime counts (Murder, Rape, Kidnapping, etc.)
- Uses a trained **Random Forest Classifier**.

### 2. Interactive Dashboard
- Displays **total crimes**, **yearly trends**, and **crime distribution**.
- Provides insights such as the most and least reported crimes.
- Visualizations built with **Matplotlib** and **Seaborn**.

---

## 📁 Project Structure
CrimePredictionApp/
│
├── app.py # Main Streamlit app
├── pages/ # Multi-page Streamlit app
│ ├── 1_Dashboard.py # Dashboard page
│ └── 2_Prediction.py # Prediction page
├── train_model.py # Script to train ML model
├── crime_prediction_model.pkl # Trained ML model
├── state_encoder.pkl # Label encoder for State
├── district_encoder.pkl # Label encoder for District
├── women-crimedataset-India.csv # Dataset
└── requirements.txt # Project dependencies

yaml
Copy code

---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/CrimePredictionApp.git
cd CrimePredictionApp
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the app locally:

bash
Copy code
streamlit run app.py
🌐 Live Demo
The app is deployed on Streamlit Cloud:
https://crimepredictionapp-k2rp8mbcownd5jdey3xifa.streamlit.app/

📊 Dependencies
streamlit

pandas

numpy

scikit-learn

matplotlib

seaborn

📈 How it Works
Dataset is preprocessed and categorical variables are encoded.

Random Forest Classifier is trained on crime data.

Users can:

Explore interactive visualizations on the dashboard.

Enter values in the prediction page to get crime level predictions.

📝 Author
Shaik Resham Afroz

BTech CSE, 3rd Year

GitHub

⚠️ Notes
Ensure women-crimedataset-India.csv is present in the project folder.

Model and encoder files are pre-trained and saved as .pkl.

yaml
Copy code

---

I can also **make a shorter, clean version optimized for GitHub portfolio** that looks professional and is easier to read.  

Do you want me to do that?







Ask ChatGPT
