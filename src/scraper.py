import yfinance as yf
import pandas as pd
import os
import logging
import argparse
from datetime import datetime, timedelta
import numpy as np
import time
from tqdm import tqdm
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_data_structure(df):
    """
    Validate that the dataframe has all the required columns.
    """
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

def fetch_stock_data(ticker, start_date, end_date, retries=3, backoff_factor=2):
    """
    Fetch historical stock data for a specific ticker from Yahoo Finance, with retry logic and exponential backoff.
    """
    for attempt in range(retries):
        try:
            logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date}, attempt {attempt + 1}")
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            if stock_data.empty:
                logger.warning(f"No data found for {ticker}. Please check if the ticker is valid.")
                return None

            stock_data.reset_index(inplace=True)
            stock_data['Ticker'] = ticker  # Add ticker column for identification
            
            validate_data_structure(stock_data)
            
            logger.info(f"Successfully fetched {len(stock_data)} rows of data for {ticker}.")
            return stock_data
        except requests.exceptions.RequestException as e:
            if "429" in str(e):  # Rate limit error
                wait_time = backoff_factor ** attempt
                logger.warning(f"Rate limit hit. Waiting for {wait_time} seconds before retrying.")
                time.sleep(wait_time)
            else:
                logger.error(f"Error fetching data for {ticker}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)  # Wait for a few seconds before retrying
                else:
                    logger.error(f"Failed to fetch data for {ticker} after {retries} attempts.")
                    return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                logger.error(f"Failed to fetch data for {ticker} after {retries} attempts.")
                return None

def calculate_technical_indicators(df):
    """
    Calculate additional technical indicators and handle NaN values
    """
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()

    # Handle NaN values
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df.fillna(method='bfill', inplace=True)  # Backward fill if necessary

    return df

def save_to_csv(df, file_path):
    """
    Save the fetched stock data to a CSV file
    """
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def scraper(tickers, start_date, end_date, output_dir):
    """
    Scraper function to fetch stock data for multiple tickers and save to CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    all_stocks_data = []
    
    # Check for existing progress
    processed_tickers = set(filename.split('_')[0] for filename in os.listdir(output_dir) if filename.endswith('_stock_data.csv'))
    tickers_to_process = [ticker for ticker in tickers if ticker not in processed_tickers]

    with tqdm(total=len(tickers), initial=len(processed_tickers), desc="Scraping stocks") as pbar:
        for ticker in tickers_to_process:
            try:
                stock_data = fetch_stock_data(ticker, start_date, end_date)
                if stock_data is not None:
                    stock_data = calculate_technical_indicators(stock_data)
                    file_path = os.path.join(output_dir, f'{ticker}_stock_data.csv')
                    save_to_csv(stock_data, file_path)
                    all_stocks_data.append(stock_data)
                time.sleep(1)  # Add a 1-second delay between API calls
            except Exception as e:
                logger.error(f"Skipping {ticker} due to an error: {e}")
            finally:
                pbar.update(1)

    if all_stocks_data:
        combined_data = pd.concat(all_stocks_data, ignore_index=True)
        combined_file_path = os.path.join(output_dir, f'all_stocks_data_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')
        save_to_csv(combined_data, combined_file_path)
        logger.info(f"Combined data saved to {combined_file_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch historical stock data")
    parser.add_argument("--tickers", nargs='+', default=['AAPL', 'GOOGL', 'MSFT'], help="List of stock tickers")
    parser.add_argument("--start_date", default=(datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'), help="Start date for stock data (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=datetime.now().strftime('%Y-%m-%d'), help="End date for stock data (YYYY-MM-DD)")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), '..', 'data'), help="Directory to save the CSV files")
    return parser.parse_args()

def validate_dates(start_date, end_date):
    """
    Validate that the start date is before the end date and the range doesn't exceed 5 years
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    if start >= end:
        raise ValueError("Start date must be before end date")
    if (end - start).days > 365 * 5:
        raise ValueError("Date range cannot exceed 5 years")

if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        validate_dates(args.start_date, args.end_date)
    except ValueError as e:
        logger.error(f"Date validation error: {e}")
        exit(1)
    
    logger.info(f"Scraping data for tickers: {args.tickers}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output directory: {args.output_dir}")

    scraper(args.tickers, args.start_date, args.end_date, args.output_dir)
    logger.info("Scraping process completed.")