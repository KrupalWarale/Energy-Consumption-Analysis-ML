"""
Energy Consumption Prediction - Multiple Regression Models Comparison
===================================================================
This script implements and compares different regression models for energy consumption prediction:
1. Linear Regression
2. Polynomial Regression
3. Random Forest Regression
4. K-Nearest Neighbors (KNN) Regression
5. Support Vector Regression (SVR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import time
import gc
import os
import pickle

# Try to import XGBoost, but handle if not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Memory optimization function
def cleanup_memory():
    gc.collect()
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
    
    # Limit dataset size if it's too large
    max_rows = 10000
    if len(df) > max_rows:
        print(f"Limiting dataset to {max_rows} rows to conserve memory")
        df = df.iloc[-max_rows:]
    
    print(f"Final dataset size: {len(df)} rows")
    return df

def prepare_features(df):
    """
    Prepare features and target variable with additional feature engineering
    """
    # Basic features
    features = ['gen [kW]', 'temperature', 'humidity', 'windSpeed', 'Solar [kW]']
    
    # Create a copy of the dataframe to avoid modifying the original
    df_features = df.copy()
    
    # Feature engineering: Add time-based features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    
    # Add interaction features
    df_features['temp_humidity'] = df_features['temperature'] * df_features['humidity']
    df_features['solar_wind'] = df_features['Solar [kW]'] * df_features['windSpeed']
    
    # Update features list with new engineered features
    features = features + ['hour', 'day_of_week', 'month', 'is_weekend', 'temp_humidity', 'solar_wind']
    
    # Extract features and target
    X = df_features[features]
    y = df_features['use [kW]']
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()  # Using StandardScaler instead of MinMaxScaler for better model performance
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Features used: {features}")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, features, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test, features):
    """
    Train and evaluate the top 3 regression models based on previous analysis
    """
    # Only include the top 3 performing models
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,  # Increased from 100 for better accuracy
            max_depth=15,      # Increased from 10 for better accuracy
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,  # Increased from 100
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ) if XGBOOST_AVAILABLE else GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        # Store results
        results[name] = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        print(f"{name} Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Training Time: {training_time:.2f} seconds")
        
        # Free memory
        gc.collect()
    
    return results

def visualize_results(results, X_test, y_test):
    """
    Visualize the performance of different models
    """
    # Create a directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # 1. Bar chart of R² scores
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    r2_scores = [results[model]['r2'] for model in models]
    
    plt.bar(models, r2_scores)
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/r2_comparison.png')
    plt.close()
    
    # 2. Bar chart of MSE
    plt.figure(figsize=(12, 6))
    mse_scores = [results[model]['mse'] for model in models]
    
    plt.bar(models, mse_scores)
    plt.title('Mean Squared Error Comparison')
    plt.ylabel('MSE')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/mse_comparison.png')
    plt.close()
    
    # 3. Actual vs Predicted for the best model
    best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
    plt.figure(figsize=(10, 6))
    
    # Get a sample of points to plot (for better visualization)
    sample_size = min(100, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    y_actual = y_test.iloc[indices].values
    y_pred = results[best_model]['predictions'][indices]
    
    plt.scatter(y_actual, y_pred, alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
    plt.title(f'Actual vs Predicted - {best_model}')
    plt.xlabel('Actual Energy Consumption (kW)')
    plt.ylabel('Predicted Energy Consumption (kW)')
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png')
    plt.close()
    
    # 4. Training time comparison
    plt.figure(figsize=(12, 6))
    training_times = [results[model]['training_time'] for model in models]
    
    plt.bar(models, training_times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/training_time_comparison.png')
    plt.close()
    
    print(f"Visualizations saved to 'plots' directory")

def make_pipeline(*steps):
    """
    Simple pipeline implementation for polynomial regression
    """
    from sklearn.pipeline import Pipeline
    return Pipeline([(f'step{i}', step) for i, step in enumerate(steps)])

def main():
    """
    Main function to run the regression model comparison
    """
    # Load and prepare data
    df = load_data(sample_size=10000)  # Use a sample for faster processing
    X_train, X_test, y_train, y_test, features, scaler = prepare_features(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, features)
    
    # Visualize results
    visualize_results(results, X_test, y_test)
    
    # Print summary
    print("\nModel Performance Summary:")
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
    print(f"\nBest performing model: {best_model}")
    print(f"R² Score: {results[best_model]['r2']:.4f}")
    print(f"MSE: {results[best_model]['mse']:.4f}")
    print(f"MAE: {results[best_model]['mae']:.4f}")
    
    # Save the best model for later use
    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/best_regression_model.pkl', 'wb') as f:
        pickle.dump({
            'model': results[best_model]['model'],
            'model_name': best_model,
            'features': features,
            'scaler': scaler,
            'metrics': {
                'r2': results[best_model]['r2'],
                'mse': results[best_model]['mse'],
                'mae': results[best_model]['mae']
            }
        }, f)
    print(f"\nBest model saved to 'models/best_regression_model.pkl'")

if __name__ == "__main__":
    main()
