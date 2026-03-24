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

        -- feature interaction
        tenure * MonthlyCharges AS CustomerValue,

        -- binary feature
        CASE
            WHEN PaymentMethod IN (
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            )
            THEN 1
            ELSE 0
        END AS AutoPay,

        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        Contract,
        PaymentMethod,

        Churn

    FROM customers_raw
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def select_features(df):

    feature_columns = [
        "tenure",
        "MonthlyCharges",
        "CustomerValue",
        "AutoPay",

        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",

        "Contract",
        "PaymentMethod",

        "Churn"
    ]

    return df[feature_columns]


def validate_data(df):

    # --- schema validation ---
    expected_columns = {
        "tenure",
        "MonthlyCharges",
        "CustomerValue",
        "AutoPay",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "Contract",
        "PaymentMethod",
        "Churn",
    }

    actual_columns = set(df.columns)

    if actual_columns != expected_columns:
        raise ValueError(
            f"Schema mismatch.\nExpected: {expected_columns}\nActual: {actual_columns}"
        )

    # --- null checks ---
    if df[["tenure", "MonthlyCharges"]].isna().any().any():
        raise ValueError("Null values detected in critical numerical columns.")

    # --- range checks ---
    if (df["tenure"] < 0).any():
        raise ValueError("Invalid value: tenure < 0 detected.")

    if (df["MonthlyCharges"] < 0).any():
        raise ValueError("Invalid value: MonthlyCharges < 0 detected.")

    # --- binary feature validation ---
    if not set(df["AutoPay"].unique()).issubset({0, 1}):
        raise ValueError("AutoPay contains invalid values.")

    # --- target validation ---
    if not set(df["Churn"].unique()).issubset({"Yes", "No"}):
        raise ValueError("Churn column contains unexpected values.")

    # --- row count guard ---
    if len(df) == 0:
        raise ValueError("Dataset is empty.")

    print("Data validation passed.")


def save_dataset(df):

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Saved dataset to {OUTPUT_PATH}")


def main():

    df = load_data()
    df = select_features(df)

    validate_data(df)

    save_dataset(df)


if __name__ == "__main__":
    main()