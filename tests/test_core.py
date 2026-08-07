import numpy as np
import pandas as pd
from src.evaluation.evaluate import compute_business_score
from src.features.build_features import FeatureBuilder

def test_business_score_perfect_prediction():
    """Test business score calculation with perfect predictions.
    All True Positives (TP) and no False Positives (FP) or False Negatives (FN).
    Expected score: 2 TP * 100 = 200.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    
    assert compute_business_score(y_true, y_pred) == 200

def test_business_score_all_wrong():
    """Test business score calculation with completely wrong predictions.
    Contains 1 False Positive (FP) and 1 False Negative (FN).
    Expected score: (0 * 100) - (1 * 75) - (1 * 200) = -275.
    """
    y_true = np.array([1, 0])
    y_pred = np.array([0, 1])
    
    assert compute_business_score(y_true, y_pred) == -275

def test_feature_builder_handles_new_category():
    """Test FeatureBuilder's robustness against unseen categorical values.
    When transforming new data with a category not seen during fit,
    the transformer should handle it via reindex and fill with 0, 
    maintaining the exact same column structure as during fit.
    """
    data = {
        'customerID': ['1'], 'gender': ['Male'], 'SeniorCitizen': [0],
        'Partner': ['No'], 'Dependents': ['No'], 'PhoneService': ['Yes'],
        'MultipleLines': ['No'], 'InternetService': ['DSL'],
        'OnlineSecurity': ['Yes'], 'OnlineBackup': ['No'],
        'DeviceProtection': ['No'], 'TechSupport': ['Yes'],
        'StreamingTV': ['No'], 'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'], 'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'], 'tenure': [12],
        'MonthlyCharges': [50.0], 'TotalCharges': ['600'], 'Churn': ['No']
    }
    df_train = pd.DataFrame(data)
    
    builder = FeatureBuilder()
    builder.fit(df_train)
    
    # Create new data with an unseen category in 'Contract'
    df_new = df_train.copy()
    df_new['Contract'] = 'Two year' 
    
    transformed = builder.transform(df_new)
    
    # Assertions: No NaN values should be present, and column count must match the fitted state
    assert not transformed.isna().any().any(), "Transformed data should not contain NaN values"
    assert transformed.shape[1] == len(builder.columns_), "Columns must align with training state"