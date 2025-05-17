import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import gc
from regression_models import load_data as load_regression_data, prepare_features, train_and_evaluate_models
from deep_learning_models import load_data as load_dl_data, prepare_sequences, train_and_evaluate_deep_models

print("Starting model pre-training...")

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
else:
    # Clean existing model files
    import glob
    for f in glob.glob('models/*.pkl'):
        os.remove(f)
    print("Cleaned existing model files")

# ================ Train Regression Models ================
print("\n======== Training Regression Models ========")
# Load a small subset of data for regression models
df_regression = load_regression_data(sample_size=2000)

# Prepare features
X_train, X_test, y_train, y_test, features, scaler = prepare_features(df_regression)

# Train and evaluate models
regression_results = train_and_evaluate_models(X_train, X_test, y_train, y_test, features)

# Select the top 3 regression models based on R²
sorted_regression_models = sorted(regression_results.items(), key=lambda x: x[1]['r2'], reverse=True)
top_3_regression_models = {name: data for name, data in sorted_regression_models[:3]}

print(f"\nSelected top 3 regression models: {', '.join(top_3_regression_models.keys())}")

# Save the top 3 regression models
for model_name, model_data in top_3_regression_models.items():
    print(f"Saving regression model: {model_name}")
    model_metadata = {
        'model': model_data['model'],
        'model_name': model_name,
        'model_type': 'regression',
        'scaler': scaler,
        'features': features,
        'metrics': {
            'mse': f"{model_data['mse']:.4f}",
            'mae': f"{model_data['mae']:.4f}",
            'r2': f"{model_data['r2']:.4f}"
        }
    }
    
    with open(f'models/regression_{model_name.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(model_metadata, f)

# ================ Train Deep Learning Models ================
print("\n======== Training Deep Learning Models ========")
# Load data for deep learning models
df_dl = load_dl_data(sample_size=2000)

# Prepare sequences
SEQ_LENGTH = 12
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, \
X_train, y_train, X_test, y_test, dl_scaler, dl_features = prepare_sequences(df_dl, seq_length=SEQ_LENGTH)

# Train and evaluate models
dl_results = train_and_evaluate_deep_models(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
                                          input_dim=X_train.shape[2], seq_length=SEQ_LENGTH)

# Select the top 3 deep learning models based on R²
sorted_dl_models = sorted(dl_results.items(), key=lambda x: x[1]['r2'], reverse=True)
top_3_dl_models = {name: data for name, data in sorted_dl_models[:3]}

print(f"\nSelected top 3 deep learning models: {', '.join(top_3_dl_models.keys())}")

# Save the top 3 deep learning models
for model_name, model_data in top_3_dl_models.items():
    print(f"Saving deep learning model: {model_name}")
    model_metadata = {
        'model': model_data['model'],
        'model_name': model_name,
        'model_type': 'deep_learning',
        'scaler': dl_scaler,
        'features': dl_features,
        'seq_length': SEQ_LENGTH,
        'metrics': {
            'mse': f"{model_data['mse']:.4f}",
            'mae': f"{model_data['mae']:.4f}",
            'r2': f"{model_data['r2']:.4f}"
        }
    }
    
    with open(f'models/dl_{model_name.replace(" ", "_").lower()}.pkl', 'wb') as f:
        pickle.dump(model_metadata, f)

# Save a models info file that includes metrics for all models for easy loading
all_models_info = {
    'regression_models': {
        name: {
            'metrics': {
                'mse': f"{data['mse']:.4f}",
                'mae': f"{data['mae']:.4f}",
                'r2': f"{data['r2']:.4f}"
            }
        } 
        for name, data in top_3_regression_models.items()
    },
    'deep_learning_models': {
        name: {
            'metrics': {
                'mse': f"{data['mse']:.4f}",
                'mae': f"{data['mae']:.4f}",
                'r2': f"{data['r2']:.4f}"
            }
        } 
        for name, data in top_3_dl_models.items()
    }
}

with open('models/all_models_info.pkl', 'wb') as f:
    pickle.dump(all_models_info, f)

# Cleanup
gc.collect()

print("\nAll 6 models (3 regression and 3 deep learning) saved successfully!")
print("You can now run the web application without retraining.") 