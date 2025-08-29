import pandas as pd
import sqlite3
import os
import logging
import argparse
import glob

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Extracting data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Extracted {len(df)} rows from {file_path}")
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Transforming data...")
    df.columns = [col.strip().lower() for col in df.columns]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError("No 'date' or 'datetime' column found")

    if 'ticker' not in df.columns:
        df['ticker'] = 'AAPL'

    if 'daily_return' not in df.columns:
        df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    if 'ma7' not in df.columns:
        df['ma7'] = df.groupby('ticker')['close'].rolling(7).mean().reset_index(0, drop=True)
    if 'ma30' not in df.columns:
        df['ma30'] = df.groupby('ticker')['close'].rolling(30).mean().reset_index(0, drop=True)
    if 'volatility' not in df.columns:
        df['volatility'] = df.groupby('ticker')['daily_return'].rolling(20).std().reset_index(0, drop=True)

    df = df[['date','ticker','open','high','low','close','volume','daily_return','ma7','ma30','volatility']]
    df = df.dropna()
    logger.info(f"Dropped rows with NaN; remaining rows: {len(df)}")
    return df

def load_data(df: pd.DataFrame, db_path: str):
    """Load data into SQLite database, recreating table fresh"""
    logger.info(f"Loading data into {db_path}...")
    df = df.copy()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Remove old database if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Deleted old database at {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            CREATE TABLE stock_data (
                Date TEXT,
                Ticker TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER,
                Daily_Return REAL,
                MA7 REAL,
                MA30 REAL,
                Volatility REAL,
                PRIMARY KEY (Date, Ticker)
            )
        ''')
        df.to_sql('stock_data', conn, if_exists='append', index=False)
    logger.info(f"Successfully loaded {len(df)} rows into 'stock_data'")

def etl_process(input_dir: str, db_file: str):
    logger.info("Starting ETL process...")
    csv_files = glob.glob(os.path.join(input_dir, '*_stock_data.csv'))
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}")
        return

    all_data = []
    for f in csv_files:
        try:
            df = extract_data(f)
            df = transform_data(df)
            all_data.append(df)
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        load_data(combined_df, db_file)
        logger.info("ETL process completed successfully!")
    else:
        logger.error("No data loaded. ETL failed.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="ETL pipeline for stock data")
    parser.add_argument("--input_dir", default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help="Directory containing raw CSV files")
    parser.add_argument("--db_file", default=os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data.db'),
                        help="Path to SQLite database")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.db_file), exist_ok=True)
    etl_process(args.input_dir, args.db_file)
