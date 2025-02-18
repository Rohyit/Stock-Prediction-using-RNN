import numpy as np
import matplotlib.pyplot as plt
import joblib  # For loading the scaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

# Function to create sequences (so we don’t need to import it from another script)
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Load stock data
def load_stock_data(ticker='AAPL'):
    df = yf.download(ticker, start="2010-01-01", end="2024-01-01", auto_adjust=True)
    return df[['Close']]

# Load and preprocess data
def load_and_preprocess_data():
    df = load_stock_data()
    scaler = joblib.load('scaler.pkl')  # Load pre-fitted MinMaxScaler
    scaled_data = scaler.transform(df)  # Transform stock prices
    return scaled_data, scaler

# Function to make predictions and plot results
def make_predictions(model, X_test, y_test_actual, scaler):
    predictions = model.predict(X_test)

    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Plot predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Compute accuracy metrics
    try:
        mse = mean_squared_error(y_test_actual, predictions)
        r2 = r2_score(y_test_actual, predictions)
        print(f"\nMean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
    except Exception as e:
        print(f"Error computing metrics: {e}")

if __name__ == "__main__":
    if __name__ == "__main__":
        print("Loading model...")
        model = load_model('model/stock_model.h5')
        print("Model loaded successfully.")

        print("Loading and preprocessing data...")
        scaled_data, scaler = load_and_preprocess_data()
        print("Data loaded and scaled.")

        print("Creating test sequences...")
        seq_length = 60
        X_test, y_test = create_sequences(scaled_data, seq_length)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        print("Making predictions...")
        print("Prediction process completed.")

        make_predictions(model, X_test, y_test, scaler)
        
