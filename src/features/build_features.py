import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "database" / "churn.db"
OUTPUT_PATH = DATA_DIR / "processed" / "features.csv"


def load_data():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        tenure,
        MonthlyCharges,
        TotalCharges,
        Contract,
        PaymentMethod,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        Churn
    FROM customers_raw
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def build_features(df):

    # 1. interaction feature
    df["CustomerValue"] = df["tenure"] * df["MonthlyCharges"]

    # 2. binary feature
    df["AutoPay"] = df["PaymentMethod"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]).astype(int)

    # 3. service count
    services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport"
    ]

    df["ServiceCount"] = (df[services] == "Yes").sum(axis=1)

    # 4. target encoding
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def encode_features(df):

    # Encode categorical to bool
    categorical_cols = [
        "Contract",
        "PaymentMethod",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Change bool dtypes to int 
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def validate_features(df):

    if df.isna().any().any():
        raise ValueError("NaN detected after feature engineering")

    if not set(df["Churn"].unique()).issubset({0, 1}):
        raise ValueError("Invalid target values")

    if len(df) == 0:
        raise ValueError("Empty dataset")


def save_data(df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


def main():
    df = load_data()
    df = build_features(df)
    df = encode_features(df)

    validate_features(df)

    save_data(df)

    print("Features pipeline completed.")


if __name__ == "__main__":
    main()