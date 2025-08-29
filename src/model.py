import numpy as np
import pandas as pd
import sqlite3
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(db_path, ticker):
    """Load data from SQLite database"""
    try:
        with sqlite3.connect(db_path) as conn:
            query = f"SELECT * FROM stock_data WHERE Ticker='{ticker}'"
            df = pd.read_sql_query(query, conn)
        logger.info(f"Successfully loaded {len(df)} rows of data")
        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        logger.error(f"Error loading data from {db_path}: {e}")
        raise

def prepare_data(df, sequence_length=60):
    """Prepare data for LSTM model with small dataset handling"""
    df = df.sort_values('Date')
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    target = df['Close'].values

    # Adjust sequence length if dataset is too small
    if len(df) <= sequence_length:
        logger.warning(f"Dataset too small, reducing sequence_length to {len(df)-1}")
        sequence_length = len(df) - 1

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(scaled_target[i])

    X, y = np.array(X), np.array(y)

    # Handle very small datasets
    if len(X) <= 1:
        logger.warning("Very small dataset: using all samples for both train and test")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler

def build_model(input_shape):
    """Build and compile LSTM model"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the LSTM model"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    return history, model

def evaluate_model(model, X_test, y_test, target_scaler):
    """Evaluate the model and print metrics"""
    y_pred = model.predict(X_test)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    accuracy = r2 * 100

    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"R-squared Score: {r2}")
    logger.info(f"Model accuracy (R-squared %): {accuracy:.2f}%")

    return y_test_inv, y_pred_inv

def plot_results(y_test, y_pred, title):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    logger.info(f"Plot saved as {title.replace(' ', '_').lower()}.png")

def predict_future(model, last_sequence, target_scaler, steps=30):
    """Predict future stock prices based on the last available sequence"""
    future_predictions = []
    
    if last_sequence.size == 0:
        logger.error("Error: No data available for future predictions.")
        return future_predictions

    try:
        current_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    except IndexError as e:
        logger.error(f"IndexError: {e}. The last sequence has incorrect dimensions: {last_sequence.shape}")
        return future_predictions

    for _ in range(steps):
        prediction = model.predict(current_sequence)
        future_predictions.append(prediction[0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = prediction[0, 0]

    future_predictions = np.array(future_predictions)
    future_predictions = target_scaler.inverse_transform(future_predictions)
    return future_predictions.flatten()

def main():
    # Arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM model for stock price prediction")
    parser.add_argument('--db_path', type=str, required=True, help="Path to SQLite database")
    parser.add_argument('--ticker', type=str, default="AAPL", help="Ticker symbol for stock data")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    # Load data
    df = load_data(args.db_path, args.ticker)

    # Prepare data
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler = prepare_data(df)

    # Build and train model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history, trained_model = train_model(model, X_train, y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch_size)

    # Evaluate model
    y_test_inv, y_pred_inv = evaluate_model(trained_model, X_test, y_test, target_scaler)
    plot_results(y_test_inv, y_pred_inv, f"Stock Price Prediction for {args.ticker}")

    # Predict future stock prices
    if X_test.size > 0:
        last_sequence = X_test[-1]
        future_pred = predict_future(trained_model, last_sequence, target_scaler)

        # Plot future predictions
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, len(future_pred)+1)]
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'][-100:], df['Close'][-100:], label='Historical Data')
        plt.plot(future_dates, future_pred, label='Future Prediction')
        plt.title(f"Future Stock Price Prediction for {args.ticker}")
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(f"future_prediction_{args.ticker}.png")
        logger.info(f"Future prediction plot saved as future_prediction_{args.ticker}.png")

if __name__ == "__main__":
    main()