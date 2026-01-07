🩺 Chronic Kidney Disease (CKD) Prediction App

📌 Project Overview

This project is a bioinformatics application developed for the course SECB3203 Programming for Bioinformatics. It utilizes Machine Learning techniques to predict the likelihood of Chronic Kidney Disease (CKD) in patients based on their clinical data (e.g., Blood Pressure, Hemoglobin, Specific Gravity).

The system is deployed as a user-friendly web application using Streamlit, allowing medical practitioners to input patient data and receive an instant diagnostic prediction with explainable AI insights.

🔗 View Live Application (Replace with your actual URL)

👥 Group Members

Group: [Insert Group Number]

Ang Chun Wei

Muhammad Haziq Bin Mohd Hafizal

Wan Nur Raudhah Binti Maszamanie

🚀 Key Features

Real-Time Prediction: Uses a trained Random Forest Classifier to predict CKD risk instantly.

Automated Data Pipeline: Fetches the latest dataset from Kaggle (kagglehub), cleans, imputes missing values, and scales features automatically on startup.

Interactive Sidebar: User-friendly form for inputting 24 clinical attributes.

Explainable AI:

Feature Importance Chart: Visualizes which medical tests contributed most to the decision.

Patient vs. Population: Compares the specific patient's values against healthy vs. sick population distributions.

🛠️ Tech Stack

Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-learn (Random Forest)

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Data Source: UCI Chronic Kidney Disease Dataset (via Kaggle)

⚙️ How to Run Locally

To run this application on your local machine, follow these steps:

Clone the Repository

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name


Install Dependencies
Ensure you have Python installed, then run:

pip install -r requirements.txt


Run the App

streamlit run app.py


📂 Project Structure

app.py: The main Streamlit application script.

requirements.txt: List of Python dependencies.

testing_guide.md: A guide with sample "Healthy" and "CKD" patient profiles for testing.

CKD_*.py: Development scripts used for EDA, Model Training, and Evaluation.

preprocessed_ckd_data.csv: The cleaned dataset used for analysis.

📊 Model Performance

Based on our project evaluation (Progress 4 & 5):

Selected Model: Random Forest Classifier

Accuracy: ~99%

AUC Score: 1.00

Cross-Validation: 0.99 (Generalized, no overfitting)

📜 Acknowledgements

Course: SECB3203 Programming for Bioinformatics

Lecturer: Dr. Sean Seah

Universiti Teknologi Malaysia (UTM)
