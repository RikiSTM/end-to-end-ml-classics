import pandas as pd
from pathlib import Path
import sqlite3
from src.data.validation import validate_raw, validate_sql_layer

# Config
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = DATA_DIR / "database" / "churn.db"


def load_raw_data():
    """
    Load raw CSV → clean → validate → save to SQLite
    """

    df = pd.read_csv(RAW_PATH)

    # Coerce numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop null
    df = df.dropna()

    # Validate
    validate_raw(df)
    print("✅ Gate 1: Pandas validation passed!")

    # Save to DB
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("customers_raw", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Data loaded ke SQLite. Rows: {len(df)}")
    
    
    validate_sql_layer()
    print("✅ Gate 2: SQL validation passed!")
    return df

if __name__ == "__main__":
    load_raw_data()