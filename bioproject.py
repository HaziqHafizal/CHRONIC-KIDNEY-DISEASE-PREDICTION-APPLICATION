import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# --- CONVERSION FUNCTIONS ---
def convert_albumin_to_grade(value_mg_dl):
    """Convert albumin mg/dL to 0-5 grade based on dipstick grading"""
    if value_mg_dl < 10: return 0      # Negative
    elif value_mg_dl < 30: return 1    # Trace
    elif value_mg_dl < 100: return 2   # 1+
    elif value_mg_dl < 300: return 3   # 2+
    elif value_mg_dl < 1000: return 4  # 3+
    else: return 5                      # 4+

def convert_sugar_to_grade(value_mg_dl):
    """Convert glucose mg/dL to 0-5 grade based on dipstick grading"""
    if value_mg_dl < 50: return 0       # Negative
    elif value_mg_dl < 150: return 1    # Trace
    elif value_mg_dl < 350: return 2    # 1+
    elif value_mg_dl < 750: return 3    # 2+
    elif value_mg_dl < 1500: return 4   # 3+
    else: return 5                       # 4+

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CKD Predictor", page_icon="🩺", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3em;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING & TRAINING FUNCTIONS ---
@st.cache_resource
def load_and_train_model():
    """
    Loads data, trains model, and returns the model + the training data 
    (so we can use the data for visualization graphs).
    """
    with st.spinner('Downloading dataset, training AI, and generating analytics...'):
        # A. Download Data
        try:
            path = kagglehub.dataset_download("mansoordaku/ckdisease")
            # Find the first CSV in the folder
            csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
            df = pd.read_csv(os.path.join(path, csv_file), na_values=['?', '\t?'])
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None, None, None

        # B. Data Cleaning
        if 'id' in df.columns:
            df.drop('id', axis=1, inplace=True)

        cols_to_force_numeric = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'sg', 'al', 'su']
        for col in cols_to_force_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Impute Missing Values
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].mean())
        for col in cat_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")

        # C. Feature Engineering (Binning Age)
        bins = [0, 40, 60, 100]
        labels = ['Young Adult', 'Middle Aged', 'Senior']
        df['Age_Group'] = pd.cut(df['age'], bins=bins, labels=labels)

        # D. Preprocessing for Model
        if 'classification' in df.columns:
            # Clean target
            df['classification'] = df['classification'].astype(str).str.strip()
            # Map to 1 (CKD) and 0 (NotCKD)
            df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0, 'ckd\t': 1}).fillna(0)
            
            X = df.drop('classification', axis=1)
            y = df['classification']
        else:
            st.error("Target column 'classification' not found.")
            return None, None, None, None, None

        # One-Hot Encoding
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Save column structure for aligning user input later
        model_columns = X_encoded.columns

        # Scaling
        scaler = MinMaxScaler()
        cols_to_scale = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        existing_cols_to_scale = [c for c in cols_to_scale if c in X_encoded.columns]
        
        X_encoded[existing_cols_to_scale] = scaler.fit_transform(X_encoded[existing_cols_to_scale])

        # E. Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_encoded, y)

        # Return the original DF too, for plotting graphs
        return model, scaler, model_columns, existing_cols_to_scale, df

# Load the model and data
model, scaler, model_columns, cols_to_scale, train_df = load_and_train_model()

# --- 2. SIDEBAR - USER INPUT ---
st.sidebar.header("📝 Patient Data Input")
st.sidebar.write("Enter the patient's details below:")

def user_input_features():
    # Group 1: Demographics & Vitals
    with st.sidebar.expander("👤 Demographics & Vitals", expanded=True):
        age = st.number_input("Age (years)", min_value=1, max_value=100, value=48)
        bp = st.number_input("Blood Pressure (mm/Hg)", min_value=50, max_value=180, value=80)
    
    # Group 2: Urinalysis
    with st.sidebar.expander("🧪 Urinalysis"):
        sg = st.number_input("Specific Gravity (ratio)", min_value=1.000, max_value=1.060, value=1.020, format="%.3f")
        al = st.number_input("Albumin (mg/dL)", min_value=0.0, max_value=2000.0, value=0.0, step=10.0)
        su = st.number_input("Sugar/Glucose (mg/dL)", min_value=0.0, max_value=3000.0, value=0.0, step=10.0)
        rbc = st.selectbox("Red Blood Cells (urine)", ["normal", "abnormal"], index=0)
        pc = st.selectbox("Pus Cell (cells/HPF)", ["normal", "abnormal"], index=0)
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"], index=0)
        ba = st.selectbox("Bacteria", ["notpresent", "present"], index=0)

    # Group 3: Blood Chemistry
    with st.sidebar.expander("🩸 Blood Chemistry"):
        bgr = st.number_input("Blood Glucose Random (mgs/dl)", value=100.0)
        bu = st.number_input("Blood Urea (mgs/dl)", value=20.0)
        sc = st.number_input("Serum Creatinine (mgs/dl)", value=1.0)
        sod = st.number_input("Sodium (mEq/L)", value=138.0)
        pot = st.number_input("Potassium (mEq/L)", value=4.5)
        hemo = st.number_input("Hemoglobin (g/dL)", value=15.0)

    # Group 4: Blood Count
    with st.sidebar.expander("🔬 Blood Count"):
        pcv = st.number_input("Packed Cell Volume (%)", value=44.0)
        wc = st.number_input("White Blood Cell Count (cells/μL)", value=7800.0)
        rc = st.number_input("Red Blood Cell Count (million/μL)", value=5.2)

    # Group 5: Disease History
    with st.sidebar.expander("📋 Medical History"):
        htn = st.selectbox("Hypertension", ["yes", "no"], index=1)
        dm = st.selectbox("Diabetes Mellitus", ["yes", "no"], index=1)
        cad = st.selectbox("Coronary Artery Disease", ["yes", "no"], index=1)
        appet = st.selectbox("Appetite", ["good", "poor"], index=0)
        pe = st.selectbox("Pedal Edema", ["yes", "no"], index=1)
        ane = st.selectbox("Anemia", ["yes", "no"], index=1)


    # Convert lab values to encoded grades for the model
    al_grade = convert_albumin_to_grade(al)
    su_grade = convert_sugar_to_grade(su)
    
    # Dictionary of inputs (using converted grades for al and su)
    data = {
        'age': age, 'bp': bp, 'sg': sg, 'al': al_grade, 'su': su_grade,
        'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
        'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
        'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba,
        'htn': htn, 'dm': dm, 'cad': cad,
        'appet': appet, 'pe': pe, 'ane': ane
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 3. MAIN PAGE DISPLAY ---
st.title("🩺 Chronic Kidney Disease Prediction")
st.markdown("""
This application uses a **Random Forest Classifier** (Accuracy: ~99%) to predict Chronic Kidney Disease.
Input patient details in the sidebar and click **Analyze** to see the diagnosis and AI explanation.
""")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Patient Data Overview")
    st.dataframe(input_df)

# --- 4. PREDICTION LOGIC ---
if st.button("Analyze Patient Data"):
    # A. Preprocessing Input
    bins = [0, 40, 60, 100]
    labels = ['Young Adult', 'Middle Aged', 'Senior']
    input_df['Age_Group'] = pd.cut(input_df['age'], bins=bins, labels=labels)

    # Encoding
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scaling
    input_encoded[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])

    # B. Prediction
    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)

    # --- 5. RESULT DISPLAY ---
    st.markdown("---")
    st.subheader("Diagnosis Result")

    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction[0] == 1:
            st.error("⚠️ **Prediction: CKD DETECTED**")
            st.write("The model predicts this patient **HAS** Chronic Kidney Disease.")
        else:
            st.success("✅ **Prediction: NO CKD DETECTED**")
            st.write("The model predicts this patient is **HEALTHY**.")

    with col_res2:
        ckd_prob = prediction_proba[0][1] * 100
        healthy_prob = prediction_proba[0][0] * 100
        st.write(f"**Confidence Score:** {max(ckd_prob, healthy_prob):.2f}%")
        
        if prediction[0] == 1:
            st.progress(int(ckd_prob))
            st.caption("Probability of CKD")
        else:
            st.progress(int(healthy_prob))
            st.caption("Probability of Healthy")

    # --- 6. EXPLAINABILITY (GRAPHS) ---
    st.markdown("---")
    st.subheader("📊 Model Insights & Findings")
    st.write("Why did the AI make this decision? See the charts below.")

    # Graph 1: Feature Importance
    st.markdown("#### 1. Top Factors Influencing the Model")
    st.write("This chart shows which medical tests the AI considers most important for diagnosis.")
    
    importances = model.feature_importances_
    feature_names = model_columns
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=forest_importances.values, y=forest_importances.index, palette="viridis", ax=ax)
    ax.set_title("Top 10 Important Features")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

    # Graph 2: Patient vs Population (Critical Indicators)
    st.markdown("#### 2. Patient vs. Population Comparison")
    st.write("The charts below show where this patient (Dashed Line) stands compared to Healthy people (Green) and CKD patients (Red).")

    # Select key numeric features to visualize
    key_features = ['hemo', 'sc', 'sg', 'al']
    feature_labels = {'hemo': 'Hemoglobin', 'sc': 'Serum Creatinine', 'sg': 'Specific Gravity', 'al': 'Albumin'}
    
    # We use the ORIGINAL training df (before scaling) to show readable values
    # We must ensure 'classification' is mapped for hue
    
    col_g1, col_g2 = st.columns(2)
    
    for i, feature in enumerate(key_features):
        with (col_g1 if i % 2 == 0 else col_g2):
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            
            # Plot distributions
            sns.kdeplot(data=train_df, x=feature, hue='classification', fill=True, 
                        palette={0: 'green', 1: 'red'}, alpha=0.3, ax=ax2)
            
            # Add Patient Line
            patient_val = input_df[feature].values[0]
            ax2.axvline(patient_val, color='blue', linestyle='--', linewidth=2, label='Current Patient')
            
            ax2.set_title(f"Distribution of {feature_labels[feature]}")
            ax2.legend(labels=['Patient', 'CKD', 'Healthy'])
            st.pyplot(fig2)
            
            # Quick Insight Text
            # Determine average for CKD vs Healthy to give a text insight
            avg_healthy = train_df[train_df['classification']==0][feature].mean()
            avg_ckd = train_df[train_df['classification']==1][feature].mean()
            
            st.caption(f"Patient {feature_labels[feature]}: **{patient_val}** (Healthy Avg: {avg_healthy:.2f}, CKD Avg: {avg_ckd:.2f})")

    # Disclaimer
    st.warning("**Note:** For educational purposes only and should not replace professional medical diagnosis.")

else:
    st.info("Awaiting input... Click 'Analyze Patient Data' to proceed.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by Group 1 for SECB3203 Bioinformatics Project")