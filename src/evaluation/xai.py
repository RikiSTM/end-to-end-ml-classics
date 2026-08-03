import shap
import matplotlib.pyplot as plt
import os
import mlflow
import numpy as np

def log_shap_to_mlflow(pipeline, X_train, run_name):
    print(f"Generating SHAP explanations for: {run_name}")
    
    # 1. Transform data
    X_train_transformed = pipeline[:-1].transform(X_train)
    
    # 2. Extract the actual trained model
    trained_model = pipeline.named_steps['model']
    
    # 3. EXTRACTION FIX: Smart Feature Name Retrieval
    try:
        # Coba ambil nama fitur langsung dari pipeline scikit-learn modern
        feature_names = pipeline[:-1].get_feature_names_out()
    except (AttributeError, ValueError, KeyError):
        try:
            # Jika preprocessor mengembalikan DataFrame
            feature_names = X_train_transformed.columns.tolist()
        except AttributeError:
            # Jika mentok jadi NumPy Array, kita pakai nama generic 
            # (Note: Untuk dapat nama asli, FeatureBuilder harus implementasi get_feature_names_out())
            feature_names = [f"feature_{i}" for i in range(X_train_transformed.shape[1])]

    # 4. PERFORMANCE OPTIMIZATION: Subsampling
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
        
        # DIMENSIONALITY FIX: Mencegah keluarnya Interaction Plot aneh
        if isinstance(raw_shap_values, list):
            # Versi SHAP lama: array ada di dalam list index 1
            shap_values = raw_shap_values[1] 
        elif len(np.array(raw_shap_values).shape) == 3:
            # Versi SHAP baru (Random Forest): Array 3D -> (n_samples, n_features, n_classes)
            # Ambil slice khusus untuk kelas 1 (Churn) agar kembali menjadi 2D
            shap_values = np.array(raw_shap_values)[:, :, 1]
        else:
            shap_values = raw_shap_values
    else:
        print(f"⚠️ SHAP logging skipped: Model '{model_class}' not supported.")
        return

    # 6. Generate and Save Plot
    plt.figure(figsize=(10, 8))
    # SHAP sekarang dipaksa menerima array 2D yang bersih
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    
    shap_filename = f"shap_summary_{run_name}.png"
    plt.tight_layout()
    plt.savefig(shap_filename, dpi=300, bbox_inches='tight')
    plt.close() # Prevent memory leaks
    
    # 7. Log to MLflow
    mlflow.log_artifact(shap_filename, artifact_path="model_explainability")
    if os.path.exists(shap_filename):
        os.remove(shap_filename)
        
    print(f"✅ SHAP Summary Plot successfully logged for {run_name}.")