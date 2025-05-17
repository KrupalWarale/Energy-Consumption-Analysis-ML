import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Load the dataset
file_path = "HomeC.csv"
df = pd.read_csv(file_path)

# Convert 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], errors='coerce', unit='s')

# Drop missing values
df = df.dropna()

# Use only the last 100,000 rows for faster training
df = df.iloc[-100000:]

# Set time as index
df.set_index('time', inplace=True)

# Selecting only 'use [kW]'
data = df[['use [kW]']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Set sequence length
sequence_length = 20  # Reduced from 50 to 20

# Prepare sequences
X, y = create_sequences(data_scaled, sequence_length)

# Split data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define an optimized LSTM model
model = Sequential([
    LSTM(25, return_sequences=False, input_shape=(sequence_length, 1)),  # Single LSTM layer with 25 neurons
    Dense(10),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train with larger batch size
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))  # Reduced epochs

# Make predictions
predicted = model.predict(X_test)

# Transform predictions back to original scale
predicted = scaler.inverse_transform(predicted)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot Actual vs Predicted Energy Consumption
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual Energy Consumption", color='blue')
plt.plot(predicted, label="Predicted Energy Consumption", color='red')
plt.xlabel("Time Steps")
plt.ylabel("Energy Consumption (kW)")
plt.title("Optimized LSTM - Actual vs Predicted Energy Consumption")
plt.legend()
plt.show()
