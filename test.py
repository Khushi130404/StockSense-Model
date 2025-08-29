# import pandas as pd
# import numpy as np
# import datetime

# # Generate dummy stock data
# def get_dummy_data(symbol="AAPL", days=30):
#     dates = pd.date_range(end=datetime.datetime.today(), periods=days)
#     data = {
#         "Open": np.random.uniform(150, 200, days),
#         "High": np.random.uniform(200, 250, days),
#         "Low": np.random.uniform(100, 150, days),
#         "Close": np.random.uniform(150, 200, days),
#         "Adj Close": np.random.uniform(150, 200, days),
#         "Volume": np.random.randint(1_000_000, 5_000_000, days),
#     }
#     df = pd.DataFrame(data, index=dates)
#     df.index.name = "Date"
#     print(f"Dummy data generated for {symbol}")
#     return df

# if __name__ == "__main__":
#     df = get_dummy_data("AAPL", days=10)
#     df.to_csv("data2.csv")   # Save to CSV
#     print("Data saved to data2.csv")

import sqlite3
import pandas as pd

conn = sqlite3.connect("data/stock_data.db")

# See tables
print(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn))

# See if stock_data exists
print(pd.read_sql("PRAGMA table_info(stock_data);", conn))

# Check tickers
print(pd.read_sql("SELECT DISTINCT Ticker FROM stock_data;", conn).head())

# Check total rows
print(pd.read_sql("SELECT COUNT(*) AS total_rows FROM stock_data;", conn))
