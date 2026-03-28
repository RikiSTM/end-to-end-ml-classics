import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.evaluation.evaluate import evaluate


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