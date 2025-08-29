import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import logging
from statsmodels.tsa.seasonal import seasonal_decompose

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_stock_data(file_path):
    """Load stock data from a CSV file."""
    try:
        logger.info(f"Loading stock data from {file_path}...")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.fillna(method='ffill', inplace=True)  # Fill missing data
        df.fillna(method='bfill', inplace=True)  # Backward fill if necessary
        logger.info(f"Successfully loaded {len(df)} rows of data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def plot_interactive_candlestick(df, ticker, output_dir):
    """Create an interactive candlestick chart using Plotly."""
    try:
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])
        fig.update_layout(title=f'{ticker} Candlestick Chart',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)

        output_path = os.path.join(output_dir, f'{ticker}_candlestick.html')
        fig.write_html(output_path)
        logger.info(f"Interactive candlestick chart saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating interactive candlestick chart for {ticker}: {e}")

def plot_advanced_technical_indicators(df, ticker, output_dir):
    """Plot advanced technical indicators including MACD, RSI, and Bollinger Bands."""
    try:
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()

        # Create subplots
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=(f'{ticker} Stock Price', 'MACD', 'RSI', 'Bollinger Bands'))

        # Price chart with Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper BB', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower BB', line=dict(dash='dash')), row=1, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal Line'), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Volume
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=4, col=1)

        fig.update_layout(height=1200, title_text=f"{ticker} Advanced Technical Analysis")
        output_path = os.path.join(output_dir, f'{ticker}_advanced_technical.html')
        fig.write_html(output_path)
        logger.info(f"Advanced technical indicators plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating advanced technical indicators plot for {ticker}: {e}")

def generate_summary_statistics(df, ticker, output_dir):
    """Generate and save summary statistics for the stock data."""
    try:
        df = df.drop(columns=['Ticker'], errors='ignore')  # Exclude 'Ticker' column from stats
        summary = df.describe()
        summary.loc['skew'] = df.skew()
        summary.loc['kurtosis'] = df.kurtosis()

        # Calculate additional metrics
        returns = df['Close'].pct_change()
        summary.loc['daily_return_mean'] = returns.mean()
        summary.loc['daily_return_std'] = returns.std()
        summary.loc['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)

        output_path = os.path.join(output_dir, f'{ticker}_summary_statistics.csv')
        summary.to_csv(output_path)
        logger.info(f"Summary statistics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating summary statistics for {ticker}: {e}")

def plot_correlation_heatmap(dfs, tickers, output_dir):
    """Plot correlation heatmap for multiple stocks."""
    try:
        close_prices = pd.DataFrame({ticker: df['Close'] for ticker, df in zip(tickers, dfs)})
        correlation = close_prices.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Stock Price Correlation Heatmap')
        
        output_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Correlation heatmap saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {e}")

def plot_seasonal_decomposition(df, ticker, output_dir):
    """Plot seasonal decomposition of the stock price."""
    try:
        decomposition = seasonal_decompose(df['Close'], model='additive', period=252)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{ticker}_seasonal_decomposition.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Seasonal decomposition plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating seasonal decomposition plot for {ticker}: {e}")

def visualize_stock_data(ticker, file_path, output_dir):
    """Generate all visualizations for a given stock and save them to the output directory."""
    try:
        df = load_stock_data(file_path)
        plot_interactive_candlestick(df, ticker, output_dir)
        plot_advanced_technical_indicators(df, ticker, output_dir)
        generate_summary_statistics(df, ticker, output_dir)
        plot_seasonal_decomposition(df, ticker, output_dir)
        return df
    except Exception as e:
        logger.error(f"Error visualizing data for {ticker}: {e}")
        return None

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize stock data")
    parser.add_argument("--tickers", nargs='+', required=True, help="List of stock tickers to visualize")
    parser.add_argument("--input_dir", default=os.path.join(os.path.dirname(__file__), '..', 'data'), help="Directory where the stock CSV files are located")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), '..', 'output'), help="Directory to save the visualizations")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Visualizing stock data for tickers: {args.tickers}")
    
    dfs = []
    for ticker in args.tickers:
        file_path = os.path.join(args.input_dir, f'{ticker}_stock_data.csv')
        if os.path.exists(file_path):
            df = visualize_stock_data(ticker, file_path, args.output_dir)
            if df is not None:
                dfs.append(df)
        else:
            logger.warning(f"Stock data file for {ticker} not found at {file_path}")

    if len(dfs) > 1:
        plot_correlation_heatmap(dfs, args.tickers, args.output_dir)

    logger.info("Visualization process completed.")