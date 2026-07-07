# 📉 Telco Customer Churn Prediction: An End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning pipeline to predict customer churn, featuring automated data ingestion, model tracking, and a containerized API for real-time inference.

---

## 1. Business Problem

Customer churn is a major challenge in the telecommunications industry. When customers cancel their subscriptions, the company loses recurring revenue. Acquiring new customers is often more expensive than retaining existing ones. 

The objective of this project is to build a machine learning model that predicts whether a customer is likely to churn. With this information, the company can take proactive actions such as:
- Targeted marketing campaigns for specific segments of customers.
- Personalized offers or discount campaigns.
- Improved service allocation.

**Impact:** Early identification of churn risk can help reduce revenue loss and improve customer lifetime value (CLV).

---

## 2. Problem Formulation

- **Task Type:** Binary Classification
- **Objective:** Predict whether a customer will churn.
- **Target Variable:** `Churn` (1 = Customer leaves, 0 = Customer stays)

The model learns patterns from historical customer data to estimate the probability of churn, serving as a decision-support system for the retention team.

---

## 3. Dataset Description

**Dataset:** Telco Customer Churn Dataset (7,043 customers)

### Feature Categories:
*   **Demographics:** Gender, SeniorCitizen, Partner, Dependents
*   **Service Info:** PhoneService, InternetService, OnlineSecurity, StreamingTV, StreamingMovies
*   **Account Info:** Contract, Tenure, MonthlyCharges, TotalCharges, PaymentMethod

---

## 4. Modeling & Evaluation Strategy

This project follows a multi-model experimentation strategy tracked via **MLflow**.

*   **Models Evaluated:** Logistic Regression (Baseline), Random Forest, XGBoost.
*   **Metrics Used:** ROC-AUC (ranking performance), Precision, Recall, F1 Score.
*   **Threshold Tuning:** Instead of using a fixed default threshold (0.5), this pipeline dynamically evaluates multiple probability thresholds and selects the one that maximizes the **F1 Score** to ensure a better balance between False Positives and False Negatives.

---

## 5. MLOps & Architecture Design

Unlike standard notebook-based projects, this repository emphasizes **production reliability**:

1.  **Experiment Tracking:** Integrated with `MLflow` to log parameters, metrics, and model artifacts seamlessly.
2.  **Modular Pipeline:** In-memory, function-based data flow (Raw Data → Validation → Feature Engineering → Train/Test Split → Model Training → Evaluation) without relying on intermediate CSVs.
3.  **Containerized Serving:** The best model is served via `FastAPI` and fully containerized using `Docker` and `docker-compose` for isolated, reproducible environments.

---

## 6. 🚀 How to Run (Docker Setup)

You can spin up the entire API and MLflow tracking server using Docker Compose.

```bash
# 1. Clone the repository
git clone [https://github.com/RikiSTM/telco-churn-mlops-pipeline.git](https://github.com/RikiSTM/telco-churn-mlops-pipeline.git)
cd telco-churn-mlops-pipeline

# 2. Build and run the containers
docker-compose up --build -d

# 3. Access the services
- FastAPI Docs (Swagger UI): http://localhost:8000/docs
- MLflow UI: http://localhost:5000
```

---

## 7. Project Structure

```text
├── app/                  # FastAPI serving logic (model_loader.py, main.py)
├── src/                  # Core pipeline scripts
│   ├── data/             # ingest.py, validation.py
│   ├── features/         # build_features.py
│   ├── models/           # train.py
│   └── evaluation/       # evaluate.py
├── data/                 # Raw and processed data storage
├── notebooks/            # EDA and initial experiments
├── docker-compose.yml    # Multi-container orchestration
├── Dockerfile            # Container configuration
└── requirements.txt      # Python dependencies
```

---

## 8. Limitations & Future Improvements
Limitations: 
- Hyperparameter tuning is not yet fully automated.
- DVC is not utilized as we are currently working with a single, static Kaggle dataset.

Future Improvements (WIP): 
- Explainable AI (XAI: Integrate tools like SHAP/LIME to debug model logic and provide interpretability for business stakeholders.
- Code Refactoring: Refactor the Great Expectations (GX) scripts using proper design patterns to improve modularity and maintainability.
- Fairness & Bias Mitigation: Incorporate Fairlearn to evaluate model fairness and mitigate potential biases in predictions.
- Comprehensive Validation: Implement Deepchecks for robust model validation, data integrity checks, and performance evaluation.

## 9. Author
Riki Sutiaman
Let's connect and discuss more about ML Engineering, MLOps, and Reliable Automation:
🔗 https://www.linkedin.com/in/riki-s-7ab291b5/
