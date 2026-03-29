# Telco Customer Churn Prediction

## 1. Business Problem

Customer churn is a major challenge in the telecommunications industry. When customers cancel their subscriptions, the company loses recurring revenue.

Acquiring new customers is often more expensive than retaining existing ones. Therefore, identifying customers who are likely to churn is critical for implementing effective retention strategies.

The objective of this project is to build a machine learning model that predicts whether a customer is likely to churn. With this information, the company can take proactive actions such as:

- Targeted marketing campaign for specific segement of the customer
- Personalized offers or discounts campaigns
- Improved service.

Early identification of churn risk can help reduce revenue loss and improve customer lifetime value.

---

## 2. Problem Formulation

The business problem is translated into a machine learning task as follows.

**Task Type**

Binary Classification

**Objective**

Predict whether a customer will churn.

**Target Variable**
Churn
1 = customer leaves
0 = customer stays


The model learns patterns from historical customer data to estimate the probability of churn.

---

## 3. Dataset Description

Dataset used: **Telco Customer Churn Dataset**

**Dataset Size**
7043 customers


**Feature Categories**

### Demographic Information

- Gender
- SeniorCitizen
- Partner
- Dependents

### Service Information

- PhoneService
- InternetService
- OnlineSecurity
- StreamingTV
- StreamingMovies

### Account Information

- Contract
- Tenure
- MonthlyCharges
- TotalCharges
- PaymentMethod

**Target Variable**
Churn


---

## 4. Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset structure and identify patterns related to churn.

Goals of EDA:

- Understand feature distributions
- Detect missing values
- Identify outliers
- Explore relationships between features and churn

---

### Dataset Overview

Initial inspection includes:

- Dataset shape 
- Data types
- Missing values
- Summary statistics
- related features

## 5. Data Pipeline

This project implements a modular data pipeline to ensure reproducibility and clear separation of concerns.

Pipeline Flow

Raw Data → Validation → Feature Engineering → Train/Test Split → Model Training → Evaluation

### Ingestion :

- Load raw dataset from CSV
- Handle data type issues (e.g. TotalCharges)
- Remove invalid or missing records
- Validate schema and data quality

### Feature Engineering :
- Create new features:
- CustomerValue = tenure × MonthlyCharges
- AutoPay indicator
- ServiceCount
- Encode categorical variables (one-hot encoding)
- Convert target variable:
- Churn: Yes → 1, No → 0

### Key Principle :
- Pipeline uses function-based data flow (in-memory)
- Avoids dependency on intermediate files (e.g. CSV as pipeline bridge)

---

## 6. Modeling Approach

This project follows a multi-model experimentation strategy.

Models Used
scikit-learn Logistic Regression (baseline)
scikit-learn Random Forest
XGBoost XGBoost
Strategy
Train multiple candidate models
Use the same dataset and preprocessing for fair comparison
Select the best model based on evaluation metrics
Why Baseline Matters

Logistic Regression is used as a baseline to:

establish a minimum performance benchmark
validate that the pipeline is working correctly

---

## 7. Evaluation Strategy

Model evaluation is not based on accuracy alone.

Metrics Used
ROC-AUC (ranking performance)
Precision
Recall
F1 Score
Threshold Tuning

Instead of using a fixed threshold (0.5), this project:

evaluates multiple thresholds
selects the threshold that maximizes F1 Score
Probability → Threshold → Final Prediction

This ensures better balance between false positives and false negatives.

---

## 8. Model Selection

All trained models are evaluated and compared.

Multiple Models → Evaluate → Select Best Model

The best model is selected based on:

highest F1 score
acceptable ROC-AUC

Only the selected model should be used for production deployment.

---

## 9. Artifacts

The following artifacts are generated:

Trained models (.pkl)
Scaler object
Evaluation results (metrics and threshold)

Artifacts are saved using joblib for reuse in inference.

---

## 10. Project Structure
src/
  ├── data/
  │     ├── ingest.py
  │     └── validation.py
  ├── features/
  │     └── build_features.py
  ├── models/
  │     └── train.py
  ├── evaluation/
  │     └── evaluate.py

data/
  ├── raw/
  ├── database/
  └── processed/

models/

---

## 11. Key Design Decisions
Single entry point (train.py) to orchestrate pipeline
Modular functions instead of script-based execution
In-memory data flow (no file dependency between steps)
Multi-model training with unified evaluation

---

## 12. Limitations
No hyperparameter tuning yet
No experiment tracking (e.g. MLflow)
No production deployment (API / batch inference)
Monitoring and drift detection not implemented

---

##  13. Future Improvements
Add experiment tracking using MLflow
Add data versioning using DVC
Deploy model using FastAPI
Implement monitoring and drift detection (e.g. Evidently AI)
Add automated pipeline orchestration (e.g. Apache Airflow)