# 📉 Telco Customer Churn Prediction: An End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Fairlearn](https://img.shields.io/badge/Fairlearn-8A2BE2?style=flat)

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
*   **Fairness & Bias Mitigation:** After finding the champion model, a post-processing mitigation is applied using `Fairlearn` to ensure demographic parity (Equalized Odds) without fundamentally altering the optimized base algorithm.
---

## 5. MLOps & Architecture Design

Unlike standard notebook-based projects, this repository emphasizes **production reliability** and **clean code architecture** (SOLID principles):

1.  **Experiment Tracking:** Integrated with `MLflow` to log parameters, metrics, and model artifacts seamlessly.
2.  **Automated Explainable AI (XAI):** Utilizes `SHAP` for dynamic feature importance extraction (supporting both Linear and Tree explainers), intercepting preprocessed data to preserve original feature names and logging visual artifacts directly to the MLflow registry.
3.  **Modular Pipeline:** In-memory, function-based data flow (Raw Data → Validation → Feature Engineering → Train/Test Split → Model Training → Evaluation) avoiding dependency on intermediate static files.
4.  **Containerized Serving:** The champion model is served via `FastAPI` and fully containerized using `Docker` and `docker-compose` for isolated, reproducible environments.
5.  **Responsible AI (Bias Mitigation):** Automatically audits the champion model for demographic bias against sensitive features (e.g., `SeniorCitizen`) and applies `ThresholdOptimizer` to correct unequal error rates before saving the artifact.

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

### 6.1. Local Training Pipeline (Development & Tracking)
To run the model training pipeline and generate MLflow experiments locally, use `poetry` for dependency management. You will need two terminal windows.

**Terminal 1: Start the MLflow Tracking Server**
```bash
# Navigate to the project root
cd telco-churn-mlops-pipeline

# Start the MLflow server on localhost
poetry run mlflow server --host 127.0.0.1 --port 5000

# Navigate to the project root
cd telco-churn-mlops-pipeline

# Run the training script as a module to resolve internal imports
poetry run python -m src.models.train
```

---

## 7. Project Structure

```text
├── app/                  # FastAPI serving logic (model_loader.py, main.py)
├── src/                  # Core pipeline scripts
│   ├── data/             # ingest.py, validation.py
│   ├── features/         # build_features.py
│   ├── models/           # train.py
│   └── evaluation/       # evaluate.py, xai.py, fairness.py (Bias Audit & Mitigation)
├── data/                 # Raw and processed data storage
├── notebooks/            # EDA and initial experiments
├── docker-compose.yml    # Multi-container orchestration
├── Dockerfile            # Container configuration
└── requirements.txt      # Python dependencies (or pyproject.toml for Poetry)
```

---

## 8. Limitations & Future Improvements
Limitations: 
- Hyperparameter tuning is not yet fully automated.
- DVC is not utilized as we are currently working with a single, static Kaggle dataset.

Future Improvements (WIP): 
- Code Refactoring: Refactor the Great Expectations (GX) scripts using proper design patterns to improve modularity and maintainability.
- Comprehensive Validation: Implement Deepchecks for robust model validation, data integrity checks, and performance evaluation.
- Add data drift check automation : Implement automatic scheduler based hit from evidently AI to pipeline for automatic drift detection

## 9. Author
Riki Sutiaman
Let's connect and discuss more about ML Engineering, MLOps, and Reliable Automation:
🔗 https://www.linkedin.com/in/riki-sutiaman-ai-engineer/

<blockquote>
  🧑‍💻 <a href="https://rikistm.github.io/">Visit my Main Portfolio</a> — RikiSTM (AI/ML Engineer | 10 Years QA Experience)</p>
</blockquote>
