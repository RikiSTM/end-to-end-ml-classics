import logging
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

# Configure standard logger for production tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_shap_to_mlflow(pipeline: Pipeline, X_train: pd.DataFrame, run_name: str) -> None:
    """
    Dynamically generates and logs SHAP summary plots to MLflow based on the model architecture.

    Args:
        pipeline (Pipeline): The trained scikit-learn pipeline containing features and the estimator.
        X_train (pd.DataFrame): The training dataset used for SHAP value computation.
        run_name (str): The name of the current MLflow run for identification.

    Returns:
        None
    """
    logger.info(f"Generating SHAP explanations for: {run_name}")
    
    # 1. Transform Data
    # Apply all pipeline steps except the final estimator to get the exact matrix the model sees
    X_train_transformed = pipeline[:-1].transform(X_train)
    
    # 2. Extract the Actual Trained Model
    trained_model = pipeline.named_steps['model']
    
    # 3. EXTRACTION FIX: Bypass StandardScaler Amnesia
    try:
        # Engineering Trick: Intercept the data right after it exits the custom FeatureBuilder,
        # strictly BEFORE it enters the StandardScaler which strips column headers.
        feature_builder = pipeline.named_steps['features']
        
        # Transform the training data ONLY through the FeatureBuilder step
        X_features_only = feature_builder.transform(X_train)
        
        # Extract the original column names directly from the resulting Pandas DataFrame
        feature_names = X_features_only.columns.tolist()
        
    except Exception as e: # noqa: BLE001
        # Fallback mechanism if the preprocessor fails to return a DataFrame
        logger.warning(f"Failed to automatically extract feature names ({e}). Using generic array indices.")
        feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

    # 4. PERFORMANCE OPTIMIZATION: Subsampling
    # Downsample to 500 rows if the dataset is large to prevent computational bottlenecks
    if X_train_transformed.shape[0] > 500:
        X_sample = shap.sample(X_train_transformed, 500)
    else:
        X_sample = X_train_transformed

    # 5. Dynamic Explainer Routing
    model_class = type(trained_model).__name__
    
    if model_class in ['LogisticRegression', 'LinearRegression']:
        explainer = shap.LinearExplainer(trained_model, X_train_transformed)
        shap_values = explainer.shap_values(X_sample)
        
    elif model_class in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
        explainer = shap.TreeExplainer(trained_model)
        raw_shap_values = explainer.shap_values(X_sample)
        
        # DIMENSIONALITY FIX: Prevent erroneous Interaction Plots
        if isinstance(raw_shap_values, list):
            # Legacy SHAP version: arrays are stored inside a list, extract the positive class
            shap_values = raw_shap_values[1] 
        elif len(np.array(raw_shap_values).shape) == 3:
            # Modern SHAP version (Random Forest): 3D Array -> (n_samples, n_features, n_classes)
            # Slice specifically for the positive class (Churn = 1) to enforce a 2D matrix
            shap_values = np.array(raw_shap_values)[:, :, 1]
        else:
            shap_values = raw_shap_values
    else:
        logger.warning(f"SHAP logging skipped: Model '{model_class}' is not supported.")
        return

    # 6. Generate and Save Plot
    plt.figure(figsize=(10, 8))
    # SHAP is now forced to accept a clean 2D array and explicit feature names
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    
    shap_filename = f"shap_summary_{run_name}.png"
    plt.tight_layout()
    plt.savefig(shap_filename, dpi=300, bbox_inches='tight')
    plt.close() # Prevent matplotlib memory leaks in production servers
    
    # 7. Log to MLflow
    mlflow.log_artifact(shap_filename, artifact_path="model_explainability")
    if os.path.exists(shap_filename):
        os.remove(shap_filename)
        
    logger.info(f"SHAP Summary Plot successfully logged for {run_name}.")