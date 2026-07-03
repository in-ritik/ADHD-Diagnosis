# ADHD Diagnostic Support System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning web app that uses a Logistic Regression model trained on Conners' Continuous Performance Test II (CPT-II) data to predict ADHD probability based on 75 selected features.

## Features

*   **Machine Learning Model**: Logistic Regression classifier trained on CPT-II features.
*   **Analysis**: Upload a patient CSV to get prediction probabilities.
*   **Visualization**: Simple Streamlit interface showing confidence metrics.
*   **Validation**: Checks predictions against actual labels if provided.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e5e93ec-be11-43e3-9e09-f553ffe6e42f" width="600"/><br/>
  <em>Interactive Streamlit Application</em>
</p>

## How It Works

This project predicts ADHD probability by looking at patient test results and demographic data.

### 1. Data Source
The model is trained on the **Conners' Continuous Performance Test II (CPT-II)**, which tests:
-   **Sustained Attention**: How well someone can keep focus.
-   **Impulsivity**: How often someone responds when they shouldn't.

### 2. Feature Selection
I used a Random Forest model to figure out which features were the most important. The top 75 features were picked, which include:
-   **Performance Metrics**: Missing targets, false alarms, and reaction time variability.
-   **Signal Processing**: Basic time series features like Fourier Transform coefficients.
-   **Demographics**: Age-standardized scores.

### 3. Prediction Model
The final model is a simple **Logistic Regression** classifier. It takes the selected features and gives a straightforward probability score.
-   **>50%**: Predicts ADHD positive.
-   **<50%**: Predicts ADHD negative.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/in-ritik/ADHD-Diagnosis.git
    cd ADHD-Diagnosis
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the app:**
    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Perform Analysis:**
    *   The interface will in browser (http://localhost:8501).
    *   The model will pre-load necessary configurations.
    *   Upload a patient CSV record from `data/patient_files/` to generate a comprehensive diagnostic report.
