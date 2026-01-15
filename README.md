# 🩺 Chronic Kidney Disease (CKD) Prediction App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-green)](https://scikit-learn.org/)

## 📌 Project Overview

This project is a bioinformatics application developed for the course **SECB3203 Programming for Bioinformatics**. It utilizes Machine Learning techniques to predict the likelihood of **Chronic Kidney Disease (CKD)** in patients based on their clinical data (e.g., Blood Pressure, Hemoglobin, Specific Gravity).

The system is deployed as a user-friendly web application using **Streamlit**, allowing medical practitioners to input patient data and receive an instant diagnostic prediction with explainable AI insights.

> **🔗 (https://chronic-kidney-disease-prediction-application.streamlit.app/)** 

---

## 👥 Group Members

**Group:** [1]

* **Ang Chun Wei**
* **Muhammad Haziq Bin Mohd Hafizal**
* **Wan Nur Raudhah Binti Maszamanie**

---

## 🚀 Key Features

* **Real-Time Prediction:** Uses a trained Random Forest Classifier to predict CKD risk instantly.
* **Automated Data Pipeline:** Fetches the latest dataset from Kaggle (`kagglehub`), cleans, imputes missing values, and scales features automatically on startup.
* **Interactive Sidebar:** User-friendly form for inputting 24 clinical attributes organized by medical category (Demographics, Urinalysis, Blood Chemistry, etc.).
* **Explainable AI:**
    * **Feature Importance Chart:** Visualizes which medical tests contributed most to the AI's decision.
    * **Patient vs. Population:** Compares the specific patient's values against "Healthy" vs. "CKD" population distributions (KDE plots) to provide clinical context.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python |
| **Web Framework** | Streamlit |
| **Machine Learning** | Scikit-learn (Random Forest) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Data Source** | UCI Chronic Kidney Disease Dataset (via Kaggle) |

---

## ⚙️ How to Run Locally

To run this application on your local machine, follow these steps:

# 1. Clone the repository (download the files)
git clone (https://github.com/HaziqHafizal/CHRONIC-KIDNEY-DISEASE-PREDICTION-APPLICATION.git)

# 2. Enter the project folder
cd CHRONIC-KIDNEY-DISEASE-PREDICTION-APPLICATION

# 3. Install the required libraries
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
