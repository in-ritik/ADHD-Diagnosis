import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.pred import load_and_merge_data, train_final_model, BEST_FEATURES, TARGET_COL

# setup
st.set_page_config(page_title="ADHD Diagnostic Support", layout="centered")

st.title("ADHD Diagnostic Support")
st.markdown("### Model Predictions")
st.divider()

# Load and train the machine learning model
def load_model():
    X, y = load_and_merge_data(feature_list=BEST_FEATURES)
    model = train_final_model(X, y)
    return model

with st.spinner("Initializing model"):
    model = load_model()

st.info("""
**About the Model**
- Uses Logistic Regression on 75 features from CPT-II and demographic data.
- Intended for educational and support purposes, not as a standalone diagnostic tool.
""")

st.subheader("Upload Patient Record")
uploaded_file = st.file_uploader("CPT-II CSV Record", type=["csv"])

if not uploaded_file:
    st.write("Please upload a patient CSV file to generate an analysis.")
else:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    
    missing_cols = [col for col in BEST_FEATURES if col not in df.columns]
    
    if missing_cols:
        st.error("Invalid file: Missing required feature columns in the uploaded CSV.")
    else:
        patient_row = df.iloc[[0]]
        features = patient_row[BEST_FEATURES]
        
        prob = float(model.predict_proba(features)[:, 1][0])
        pred = int(model.predict(features)[0])
        
        st.divider()
        st.subheader("Diagnostic Results")
        
        if pred == 1:
            st.error("ADHD POSITIVE")
        else:
            st.success("ADHD NEGATIVE")
            
        st.metric("Probability of ADHD", f"{prob:.1%}")
        st.progress(prob, text="Prediction Confidence")
        
        # Ground truth validation
        if TARGET_COL in patient_row.columns:
            actual = int(patient_row[TARGET_COL].values[0])
            correct = (pred == actual)
            label_text = "Positive" if actual == 1 else "Negative"
            
            st.divider()
            if correct:
                st.success(f"Matched Ground Truth ({label_text})")
            else:
                st.warning(f"Did not match Ground Truth ({label_text})")
