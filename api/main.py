import numpy as np
import pandas as pd
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


# --- Pydantic Models: Defines the shape of our JSON output ---
class ChartDataPoint(BaseModel):
    date: str
    actual: float
    predicted: float
    signal: Optional[str] = None # Will be 'buy', 'sell', or null

class ChartResponse(BaseModel):
    chartData: List[ChartDataPoint]

# --- FastAPI App Setup ---
app = FastAPI(
    title="StockSense Signal API",
    description="Provides chart data with actionable buy/sell signals."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your React app to connect
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_signals_and_data(ticker: str) -> pd.DataFrame:
    """
    This is the main "decision engine". It creates mock data and generates signals.
    """
    # 1. Generate realistic-looking mock data
    logger.info(f"Generating mock data for {ticker}...")
    dates = pd.to_datetime(pd.date_range(end='2025-08-29', periods=100))
    # Create a base sine wave with some noise for realism
    actual_prices = 150 + np.sin(np.arange(100) * 0.2) * 10 + np.random.randn(100) * 2
    df = pd.DataFrame({'date': dates, 'actual': actual_prices})
    # Mock predicted price is a smoothed version of the actual price
    df['predicted'] = df['actual'].rolling(window=5, min_periods=1).mean()

    # 2. Detect "Drastic Changes" (Volatility Spikes)
    df['daily_return'] = df['actual'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=20).std()
    # A drastic change is a move > 2x the recent volatility
    df['is_drastic'] = abs(df['daily_return']) > (2.0 * df['volatility'])

    # 3. MOCK Sentiment Analysis
    # In your real app, this would come from your sentiment model.
    # Here, we'll simulate it for the days with drastic changes.
    sentiments = []
    for is_drastic in df['is_drastic']:
        if is_drastic:
            sentiments.append('positive' if np.random.rand() > 0.5 else 'negative')
        else:
            sentiments.append('neutral')
    df['sentiment'] = sentiments

    # 4. The Decision Matrix: Fuse signals to decide buy/sell/hold
    conditions = [
        (df['is_drastic'] == True) & (df['sentiment'] == 'positive'), # Drastic move + good news
        (df['is_drastic'] == True) & (df['sentiment'] == 'negative')  # Drastic move + bad news
    ]
    choices = ['buy', 'sell']
    df['signal'] = np.select(conditions, choices, default='hold')
    
    return df



# --- The API Endpoint ---



@app.get("/forecast_chart_data/{ticker}", response_model=ChartResponse)
def getForecastChartDataEndpoint(ticker: str):
    """
    API endpoint that generates and returns the chart data with signals.
    """
    df_with_signals = generate_signals_and_data(ticker)
    
    # Format the data into the structure our Pydantic model expects
    chart_data_list = []
    for _, row in df_with_signals.iterrows():
        # Only include the signal if it's a 'buy' or 'sell'
        signal = row['signal'] if row['signal'] in ['buy', 'sell'] else None
        chart_data_list.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "actual": round(row['actual'], 2),
            "predicted": round(row['predicted'], 2),
            "signal": signal
        })
        
    return {"chartData": chart_data_list}

# Pydantic model for chart data without signals
class PredictionDataPoint(BaseModel):
    date: str
    actual: float
    predicted: float

class PredictionResponse(BaseModel):
    data: List[PredictionDataPoint]

@app.get("/predict_chart_data/{ticker}", response_model=PredictionResponse)
def getPredictionChartData(ticker: str):
    """
    Returns actual vs predicted prices in JSON format.
    """
    df = generate_signals_and_data(ticker)
    
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "actual": round(row['actual'], 2),
            "predicted": round(row['predicted'], 2)
        })
    
    return {"data": data_list}
