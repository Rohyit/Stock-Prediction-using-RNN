import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib  # Import joblib for saving/loading the scaler

# Function to download and preprocess data
def load_and_preprocess_data(stock_symbol='AAPL', start_date='2010-01-01', end_date='2020-12-31'):
    # Download stock data from Yahoo Finance
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    # Show first 5 rows to inspect the data
    print(data.head())

    # Use only the 'Close' price for prediction
    data = data[['Close']]

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Save the scaler to a file using joblib
    joblib.dump(scaler, 'scaler.pkl')

    return scaled_data, scaler, data

# Save the processed data to a CSV file (optional)
def save_data(data, filename="AAPL_stock_data.csv"):
    data.to_csv(filename)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Preprocess the data
    scaled_data, scaler, original_data = load_and_preprocess_data()

    # Optionally, save the raw data to a CSV
    save_data(original_data)
