import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Function to build and train the LSTM model
def build_and_train_model(X_train, y_train, seq_length=100):
    # Build the LSTM model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))  # Predict the next day's closing price

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Save the model to a file (optional)
    model.save('model/stock_model.h5')

    return model

if __name__ == "__main__":
    # Load preprocessed data (X_train, y_train, X_test, y_test)
    X_train = np.load('X_data.npy')
    y_train = np.load('y_data.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Reshape the data to be [samples, time steps, features] (as required for LSTM)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Evaluate the model (optional)
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
