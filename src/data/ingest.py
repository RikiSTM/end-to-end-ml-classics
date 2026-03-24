import pandas as pd
from pathlib import Path
import sqlite3
from src.data.validation import validate_raw

# Config 
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_PATH = DATA_DIR /"database"/ "churn.db"

# Load & simpan ke SQLite
df = pd.read_csv(RAW_PATH)

# Checl for coerce
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df = df.dropna()

# Call validation to check
validate_raw(df)

conn = sqlite3.connect(DB_PATH)
df.to_sql("customers_raw", conn, if_exists="replace", index=False)
conn.close()

print(f"✅ Data loaded ke SQLite! Total rows: {len(df)}")
print(f"DB ada di: {DB_PATH}")