"""
Energy Consumption Prediction - Deep Learning Models
===================================================
This script implements deep learning models for energy consumption prediction:
1. LSTM (Long Short-Term Memory) Network
2. GRU (Gated Recurrent Unit) Network
3. 1D CNN (Convolutional Neural Network)
4. Hybrid CNN-LSTM Network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import gc
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Memory optimization function
def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("Memory cleanup performed")

def load_data(file_path="HomeC.csv", sample_size=None):
    """
    Load and preprocess the energy consumption dataset
    """
    print("Loading dataset...")
    
    # Define the columns we need to reduce memory usage
    usecols = ['time', 'use [kW]', 'gen [kW]', 'temperature', 'humidity', 'windSpeed', 'Solar [kW]']
    
    # Load data with optional sample size for testing
    if sample_size:
        df = pd.read_csv(file_path, usecols=usecols, nrows=sample_size)
    else:
        # For production, use a more memory-efficient approach with chunks
        chunk_size = 10000
        total_chunks = 5  # Limit to 5 chunks to avoid memory issues
        
        df = pd.DataFrame()
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, usecols=usecols)):
            if i >= total_chunks:
                break
            df = pd.concat([df, chunk])
            del chunk
            gc.collect()
    
    # Preprocess the data
    print("Preprocessing data...")
    df = df[pd.to_numeric(df['time'], errors='coerce').notna()]
    df['time'] = df['time'].astype(float).astype(int)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.dropna(inplace=True)
    
    # Sort by time index
    df = df.sort_index()
    
    # Limit dataset size if it's too large
    max_rows = 20000  # Increased for deep learning
    if len(df) > max_rows:
        print(f"Limiting dataset to {max_rows} rows to conserve memory")
        df = df.iloc[-max_rows:]
    
    print(f"Final dataset size: {len(df)} rows")
    return df

def prepare_sequences(df, seq_length=24):
    """
    Prepare time series sequences for deep learning models
    """
    print(f"Creating sequences with length {seq_length}...")
    
    # Feature engineering
    df_features = df.copy()
    
    # Add time-based features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    
    # Add interaction features
    df_features['temp_humidity'] = df_features['temperature'] * df_features['humidity']
    df_features['solar_wind'] = df_features['Solar [kW]'] * df_features['windSpeed']
    
    # Define features
    features = ['use [kW]', 'gen [kW]', 'temperature', 'humidity', 'windSpeed', 'Solar [kW]',
                'hour', 'day_of_week', 'month', 'is_weekend', 'temp_humidity', 'solar_wind']
    
    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_features[features]),
        columns=features,
        index=df_features.index
    )
    
    # Create sequences
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled.iloc[i:i+seq_length].values)
        y.append(df_scaled.iloc[i+seq_length]['use [kW]'])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle for time series
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            X_train, y_train, X_test, y_test, scaler, features)

# Define neural network models
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class CNNModel(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim):
        super(CNNModel, self).__init__()
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after pooling
        self.flatten_size = 128 * (seq_length // 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape input: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class HybridCNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(HybridCNNLSTMModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            32, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Reshape for CNN: [batch, seq_len, features] -> [batch, features, seq_len]
        x_cnn = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x_cnn = self.relu(self.conv1(x_cnn))
        x_cnn = self.relu(self.conv2(x_cnn))
        
        # Reshape back for LSTM: [batch, features, seq_len] -> [batch, seq_len, features]
        x_lstm = x_cnn.permute(0, 2, 1)
        
        # Apply LSTM
        out, _ = self.lstm(x_lstm)
        
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    """
    Train a PyTorch model
    """
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return train_losses

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model and return performance metrics
    """
    model.eval()
    
    predictions = []
    actuals = []
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            
            # Accumulate loss
            total_loss += loss.item() * inputs.size(0)
            
            # Store predictions and actuals
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(targets.squeeze().cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return mse, mae, r2, predictions

def train_and_evaluate_deep_models(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim, seq_length):
    """
    Train and evaluate deep learning models
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model parameters
    batch_size = 32
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    epochs = 50
    
    # Define the models
    models = {
        'LSTM': LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device),
        'GRU': GRUModel(input_dim, hidden_dim, num_layers, output_dim).to(device),
        'CNN': CNNModel(input_dim, seq_length, output_dim).to(device)
    }
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        start_time = time.time()
        
        # Define optimizer for this model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        train_losses = train_model(model, train_loader, criterion, optimizer, device, epochs=epochs)
        
        # Evaluate the model
        mse, mae, r2, y_pred = evaluate_model(model, test_loader, criterion, device)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time,
            'train_losses': train_losses
        }
        
        print(f"{name} Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Training Time: {training_time:.2f} seconds")
        
        # Clean up memory
        cleanup_memory()
    
    return results

def visualize_results(results, seq_length):
    """
    Visualize the performance of different models
    """
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots/deep_learning', exist_ok=True)
    
    # 1. Bar chart of R² scores
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    r2_scores = [results[model]['r2'] for model in models]
    
    plt.bar(models, r2_scores)
    plt.title('Deep Learning Models - R² Score Comparison')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/deep_learning/r2_comparison.png')
    plt.close()
    
    # 2. Bar chart of MSE
    plt.figure(figsize=(12, 6))
    mse_scores = [results[model]['mse'] for model in models]
    
    plt.bar(models, mse_scores)
    plt.title('Deep Learning Models - Mean Squared Error Comparison')
    plt.ylabel('MSE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/deep_learning/mse_comparison.png')
    plt.close()
    
    # 3. Training time comparison
    plt.figure(figsize=(12, 6))
    training_times = [results[model]['training_time'] for model in models]
    
    plt.bar(models, training_times)
    plt.title('Deep Learning Models - Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/deep_learning/training_time_comparison.png')
    plt.close()
    
    # 4. Training loss curves
    plt.figure(figsize=(12, 6))
    for name, metrics in results.items():
        plt.plot(metrics['train_losses'], label=name)
    
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/deep_learning/training_loss_curves.png')
    plt.close()
    
    # 5. Actual vs Predicted for the best model
    best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
    plt.figure(figsize=(10, 6))
    
    # Get a sample of points to plot (for better visualization)
    sample_size = min(100, len(results[best_model]['actuals']))
    indices = np.random.choice(len(results[best_model]['actuals']), sample_size, replace=False)
    
    y_actual = results[best_model]['actuals'][indices].flatten()
    y_pred = results[best_model]['predictions'][indices].flatten()
    
    plt.scatter(y_actual, y_pred, alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    plt.title(f'Actual vs Predicted - {best_model}')
    plt.xlabel('Actual Energy Consumption (Normalized)')
    plt.ylabel('Predicted Energy Consumption (Normalized)')
    plt.tight_layout()
    plt.savefig('plots/deep_learning/actual_vs_predicted.png')
    plt.close()
    
    print(f"Visualizations saved to 'plots/deep_learning' directory")

def main():
    """
    Main function to run the deep learning model comparison
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    df = load_data(sample_size=20000)  # Use a larger sample for deep learning
    
    # Prepare sequences
    seq_length = 24  # Use 24 hours of data to predict the next hour
    (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
     X_train, y_train, X_test, y_test, scaler, features) = prepare_sequences(df, seq_length)
    
    # Train and evaluate models
    input_dim = X_train.shape[2]  # Number of features
    results = train_and_evaluate_deep_models(
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim, seq_length
    )
    
    # Visualize results
    visualize_results(results, seq_length)
    
    # Print summary
    print("\nDeep Learning Model Performance Summary:")
    print("=" * 50)
    
    # Sort models by R² score for better readability
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True))
    
    for name, metrics in sorted_results.items():
        print(f"{name}:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print("-" * 50)
    
    # Identify the best model
    best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"\nBest performing deep learning model: {best_model}")
    print(f"R² Score: {results[best_model]['r2']:.4f}")
    print(f"MSE: {results[best_model]['mse']:.4f}")
    print(f"MAE: {results[best_model]['mae']:.4f}")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': results[best_model]['model'].state_dict(),
        'model_name': best_model,
        'input_dim': input_dim,
        'seq_length': seq_length,
        'features': features,
        'scaler': scaler,
        'metrics': {
            'r2': results[best_model]['r2'],
            'mse': results[best_model]['mse'],
            'mae': results[best_model]['mae']
        }
    }, 'models/best_deep_learning_model.pth')
    
    print(f"\nBest model saved to 'models/best_deep_learning_model.pth'")

if __name__ == "__main__":
    main()
