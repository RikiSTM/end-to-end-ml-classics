import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
FEATURE_PATH = DATA_DIR / "processed" / "features.csv"

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "logreg.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"



def load_data():
    df = pd.read_csv(FEATURE_PATH)
    return df


def split_data(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")

    model.fit(X_train_scaled, y_train)

    return model,scaler


def evaluate(model,scaler, X_test, y_test):

    X_test_scaled = scaler.transform(X_test)    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Confusion Matrix: ")
    print(cm)

def save_artifacts(model, scaler):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


def main():

    df = load_data()
    
    print(df.shape)
    print(df["Churn"].value_counts(normalize=True))

    X_train, X_test, y_train, y_test = split_data(df)

    model, scaler = train_model(X_train, y_train)

    evaluate(model, scaler, X_test, y_test)

    save_artifacts(model, scaler)


if __name__ == "__main__":
    main()