import sqlite3
import pandas as pd
import os

db_path = os.path.join("data", "stock_data.db")
conn = sqlite3.connect(db_path)

# Ensure db directory exists
os.makedirs("data", exist_ok=True)

# Load all processed CSVs
processed_dir = "processed"
csv_files = [f for f in os.listdir(processed_dir) if f.endswith("_stock_data.csv")]

for file in csv_files:
    path = os.path.join(processed_dir, file)
    df = pd.read_csv(path)

    # Normalize column names to match model.py expectations
    df.columns = [c.capitalize() for c in df.columns]

    df.to_sql("stock_data", conn, if_exists="append", index=False)
    print(f"✅ Loaded {len(df)} rows from {file} into stock_data table")

conn.close()
