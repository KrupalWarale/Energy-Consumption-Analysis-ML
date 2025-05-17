from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
import os
import pickle
import gc  # Garbage collector
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
import glob
import datetime

# Try to import XGBoost if available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

app = Flask(__name__)

# Global variables for each model
models = {}
predictions = {}
all_models_info = None
time_index = None
actual_values = None

# Check if models exist
models_exist = False
models_path = 'models'
if os.path.exists(models_path):
    model_files = glob.glob(os.path.join(models_path, '*.pkl'))
    if len(model_files) > 0:
        models_exist = True

# Load all models if they exist
if models_exist:
    try:
        # First try to load the all_models_info file which contains metadata about all models
        if os.path.exists('models/all_models_info.pkl'):
            with open('models/all_models_info.pkl', 'rb') as f:
                all_models_info = pickle.load(f)
            print("Loaded models info successfully")
        
        # Load regression models
        regression_model_files = glob.glob('models/regression_*.pkl')
        for model_file in regression_model_files:
            model_name = os.path.basename(model_file).replace('.pkl', '').replace('regression_', '')
            model_name = model_name.replace('_', ' ').title()  # Convert to readable format
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            models[model_name] = {
                'model': model_data['model'],
                'model_type': 'regression',
                'scaler': model_data['scaler'],
                'features': model_data['features'],
                'metrics': model_data['metrics']
            }
            print(f"Loaded regression model: {model_name}")
            print(f"Model metrics: R²={models[model_name]['metrics']['r2']}, MSE={models[model_name]['metrics']['mse']}")
        
        # Load deep learning models
        dl_model_files = glob.glob('models/dl_*.pkl')
        for model_file in dl_model_files:
            model_name = os.path.basename(model_file).replace('.pkl', '').replace('dl_', '')
            model_name = model_name.replace('_', ' ').title()  # Convert to readable format
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            models[model_name] = {
                'model': model_data['model'],
                'model_type': 'deep_learning',
                'scaler': model_data['scaler'],
                'features': model_data['features'],
                'metrics': model_data['metrics']
            }
            
            if 'seq_length' in model_data:
                models[model_name]['seq_length'] = model_data['seq_length']
                
            print(f"Loaded deep learning model: {model_name}")
            print(f"Model metrics: R²={models[model_name]['metrics']['r2']}, MSE={models[model_name]['metrics']['mse']}")
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        models_exist = False

@app.route('/')
def index():
    return render_template('index.html', models=models, models_exist=models_exist, models_info=all_models_info)

@app.route('/api/data')
def api_data():
    global predictions, actual_values, time_index
    
    try:
        # Check if we have predictions
        if not predictions or len(predictions) == 0:
            return jsonify({"error": "No predictions available. Please run predictions first."}), 400
            
        if actual_values is None or time_index is None:
            return jsonify({"error": "No actual values or timestamps available"}), 400
            
        # Get the model name from the request, if any
        model_name = request.args.get('model', None)
        
        # Convert timestamps to string format for JSON
        timestamps = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in time_index]
        
        # Subsample data if too large (> 1000 points)
        if len(timestamps) > 1000:
            # Calculate step to sample ~1000 points
            step = max(1, len(timestamps) // 1000)
            timestamps = timestamps[::step]
            actual = actual_values[::step].tolist()
        else:
            actual = actual_values.tolist()
            
        # Prepare the response
        response = {
            "timestamps": timestamps,
            "actual": actual
        }
        
        # If a specific model is requested
        if model_name:
            if model_name not in predictions:
                return jsonify({"error": f"Model '{model_name}' not found in predictions"}), 404
                
            # Subsample predictions if needed
            if len(predictions[model_name]) > 1000:
                step = max(1, len(predictions[model_name]) // 1000)
                pred_data = predictions[model_name][::step].tolist()
            else:
                pred_data = predictions[model_name].tolist()
                
            # Check for NaNs and replace with nulls for JSON
            pred_data = [None if np.isnan(x) else x for x in pred_data]
            
            # Add to response for single model format
            response["predicted"] = pred_data
        else:
            # Return all models' predictions
            response["predictions"] = {}
            
            for name, preds in predictions.items():
                # Subsample predictions if needed
                if len(preds) > 1000:
                    step = max(1, len(preds) // 1000)
                    pred_data = preds[::step].tolist()
                else:
                    pred_data = preds.tolist()
                    
                # Check for NaNs and replace with nulls for JSON
                pred_data = [None if np.isnan(x) else x for x in pred_data]
                    
                # Make sure lengths match timestamps after subsampling
                if len(pred_data) > len(timestamps):
                    pred_data = pred_data[:len(timestamps)]
                elif len(pred_data) < len(timestamps):
                    # Pad with nulls
                    pred_data.extend([None] * (len(timestamps) - len(pred_data)))
                
                response["predictions"][name] = pred_data
                
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in API data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/predict')
def predict():
    global predictions, actual_values, time_index
    
    if not models_exist:
        return jsonify({"error": "No pre-trained models available. Please run train_and_save_model.py first."}), 400
    
    try:
        # Load a sample of data for prediction
        print("Loading test data for prediction...")
        sample_size = 500  # Use a larger sample for better visualization
        
        # Load the CSV file
        df = pd.read_csv("HomeC.csv", nrows=sample_size)
        
        # Preprocess the data
        print("Preprocessing data...")
        df = df[pd.to_numeric(df['time'], errors='coerce').notna()]
        df['time'] = df['time'].astype(float).astype(int)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.dropna(inplace=True)
        
        # Store actual values and time index for plotting
        actual_values = df['use [kW]'].values
        time_index = df.index
        
        # Generate predictions for each model
        predictions = {}
        
        for model_name, model_data in models.items():
            try:
                print(f"Generating predictions for {model_name}...")
                model = model_data['model']
                model_type = model_data['model_type']
                scaler = model_data['scaler']
                features = model_data['features']
                
                # Process based on model type
                if model_type == 'regression':
                    # Feature engineering (same as in training)
                    df_features = df.copy()
                    
                    # Add time-based features
                    df_features['hour'] = df_features.index.hour
                    df_features['day_of_week'] = df_features.index.dayofweek
                    df_features['month'] = df_features.index.month
                    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
                    
                    # Add interaction features
                    df_features['temp_humidity'] = df_features['temperature'] * df_features['humidity']
                    df_features['solar_wind'] = df_features['Solar [kW]'] * df_features['windSpeed']
                    
                    # Extract features
                    X = df_features[features]
                    
                    # Handle missing values if any
                    X = X.fillna(X.mean())
                    
                    # Scale features
                    X_scaled = scaler.transform(X)
                    
                    # Generate predictions
                    model_predictions = model.predict(X_scaled)
                    
                    # Ensure predictions has the same length as actual_values
                    if len(model_predictions) != len(actual_values):
                        print(f"Warning: {model_name} predictions length ({len(model_predictions)}) doesn't match actual values length ({len(actual_values)})")
                        # Pad or truncate as needed
                        if len(model_predictions) > len(actual_values):
                            model_predictions = model_predictions[:len(actual_values)]
                        else:
                            # Pad with NaN
                            padding = np.full(len(actual_values) - len(model_predictions), np.nan)
                            model_predictions = np.concatenate([model_predictions, padding])
                    
                    predictions[model_name] = model_predictions
                    
                elif model_type == 'deep_learning':
                    # For deep learning models, we need to create sequences
                    seq_length = model_data.get('seq_length', 12)  # Default to 12 if not specified
                    
                    # Create a copy for feature engineering
                    df_features = df.copy()
                    
                    # Add time-based features
                    df_features['hour'] = df_features.index.hour
                    df_features['day_of_week'] = df_features.index.dayofweek
                    df_features['month'] = df_features.index.month
                    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
                    
                    # Add interaction features
                    df_features['temp_humidity'] = df_features['temperature'] * df_features['humidity']
                    df_features['solar_wind'] = df_features['Solar [kW]'] * df_features['windSpeed']
                    
                    # Make sure all features are available
                    for feature in features:
                        if feature not in df_features.columns:
                            print(f"Warning: Feature '{feature}' not in dataframe. Adding it with zeros.")
                            df_features[feature] = 0
                    
                    # Scale features
                    df_scaled = pd.DataFrame(
                        scaler.transform(df_features[features]),
                        columns=features,
                        index=df_features.index
                    )
                    
                    # Create sequences
                    X_sequences = []
                    for i in range(len(df_scaled) - seq_length):
                        X_sequences.append(df_scaled.iloc[i:i+seq_length].values)
                    
                    # Check if we have sequences
                    if len(X_sequences) == 0:
                        print(f"Warning: No sequences could be created for {model_name}")
                        # Create dummy predictions with NaN
                        predictions[model_name] = np.full(len(actual_values), np.nan)
                        continue
                    
                    # Convert to torch tensor
                    X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)
                    
                    # Set model to evaluation mode
                    model.eval()
                    
                    # Generate predictions
                    with torch.no_grad():
                        dl_predictions = model(X_tensor).squeeze().numpy()
                    
                    # Convert predictions back to original scale if needed
                    if isinstance(dl_predictions, np.ndarray) and len(dl_predictions.shape) > 0:
                        if len(dl_predictions.shape) == 1:
                            # Handle case where predictions are 1D
                            try:
                                # Use target index
                                target_idx = features.index('use [kW]')
                                
                                # Create a placeholder array with all zeros except for the predictions
                                dl_predictions_reshaped = np.zeros((len(dl_predictions), len(features)))
                                dl_predictions_reshaped[:, target_idx] = dl_predictions
                                
                                # Inverse transform to original scale
                                dl_predictions_original = scaler.inverse_transform(dl_predictions_reshaped)[:, target_idx]
                            except Exception as e:
                                print(f"Error scaling predictions: {str(e)}")
                                dl_predictions_original = dl_predictions  # Use as-is if scaling fails
                        else:
                            # Handle case where predictions might be multi-dimensional
                            try:
                                target_idx = features.index('use [kW]')
                                dl_predictions_original = scaler.inverse_transform(dl_predictions)[:, target_idx]
                            except Exception as e:
                                print(f"Error scaling multi-dim predictions: {str(e)}")
                                dl_predictions_original = dl_predictions.mean(axis=1) if len(dl_predictions.shape) > 1 else dl_predictions
                    else:
                        # Handle scalar output or empty array
                        print(f"Warning: Unexpected prediction format from {model_name}")
                        dl_predictions_original = np.array([dl_predictions] if np.isscalar(dl_predictions) else [])
                    
                    # Store predictions but pad with NaN for the sequence length
                    full_predictions = np.full(len(df), np.nan)
                    
                    # Handle the case where predictions might be longer than expected
                    pred_length = min(len(dl_predictions_original), len(full_predictions) - seq_length)
                    if pred_length > 0:
                        full_predictions[seq_length:seq_length+pred_length] = dl_predictions_original[:pred_length]
                    
                    predictions[model_name] = full_predictions
            except Exception as e:
                print(f"Error generating predictions for {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Add empty predictions for this model
                predictions[model_name] = np.full(len(actual_values), np.nan)
        
        return jsonify({"status": "success"})
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/plot')
def get_plot():
    model_name = request.args.get('model', '')
    
    if not predictions:
        return jsonify({"error": "No predictions available"})
    
    # Improved plot styling
    plt.style.use('seaborn-v0_8-pastel')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = 0.3
    
    if model_name and model_name in predictions:
        # Single model plot
        model_predictions = predictions[model_name]
        
        # Generate plot with a subset of data to save memory
        max_points = 150  # Increased from 100 for smoother plots
        if len(actual_values) > max_points:
            step = len(actual_values) // max_points
            indices = np.arange(0, len(actual_values), step)
            plot_time = [time_index[i] for i in indices]
            plot_actual = [actual_values[i] for i in indices]
            plot_pred = [model_predictions[i] for i in indices]
        else:
            plot_time = time_index
            plot_actual = actual_values
            plot_pred = model_predictions
        
        # Generate enhanced plot
        plt.figure(figsize=(12, 6), dpi=100)  # Increased for better quality
        
        # Add a light grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot the data with improved styling
        plt.plot(plot_time, plot_actual, label="Actual Energy Consumption", 
                 color='#3498db', linewidth=2.5, marker='o', markersize=4, markevery=int(len(plot_time)/20))
        
        plt.plot(plot_time, plot_pred, label=f"{model_name} Prediction", 
                 color='#e74c3c', linewidth=2, linestyle='--', marker='s', 
                 markersize=4, markevery=int(len(plot_time)/20))
        
        # Add a light fill below the actual consumption line
        plt.fill_between(plot_time, plot_actual, alpha=0.1, color='#3498db')
        
        # Better styling for legend, title and labels
        plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        plt.title(f"Actual vs {model_name} Predicted Energy Consumption", 
                  fontsize=14, fontweight='bold', pad=20)
        
        plt.xlabel("Time", fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel("Energy Consumption (kW)", fontsize=12, fontweight='bold', labelpad=10)
        
        # Improve the appearance of the x-axis
        plt.tick_params(axis='x', labelrotation=30)
        plt.tight_layout()
        
    else:
        # Generate a plot with all models
        # Limit to at most 3 models to avoid cluttering
        models_to_plot = list(predictions.keys())[:3]
        
        # Generate plot with a subset of data to save memory
        max_points = 150  # Increased from 100 for smoother plots
        if len(actual_values) > max_points:
            step = len(actual_values) // max_points
            indices = np.arange(0, len(actual_values), step)
            plot_time = [time_index[i] for i in indices]
            plot_actual = [actual_values[i] for i in indices]
            plot_preds = {model: [predictions[model][i] for i in indices] for model in models_to_plot}
        else:
            plot_time = time_index
            plot_actual = actual_values
            plot_preds = {model: predictions[model] for model in models_to_plot}
        
        # Generate enhanced plot
        plt.figure(figsize=(12, 6), dpi=100)  # Increased for better quality
        
        # Add a light grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Different line styles and colors for different models
        colors = ['#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
        line_styles = ['--', '-.', ':']
        markers = ['s', '^', 'd']
        
        # Plot actual values with prominent styling
        plt.plot(plot_time, plot_actual, label="Actual Energy Consumption", 
                 color='#3498db', linewidth=2.5, marker='o', markersize=4, markevery=int(len(plot_time)/20))
        
        # Add a light fill below the actual consumption line
        plt.fill_between(plot_time, plot_actual, alpha=0.1, color='#3498db')
        
        # Plot each model's predictions
        for i, model in enumerate(models_to_plot):
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            
            plt.plot(plot_time, plot_preds[model], label=f"{model} Prediction", 
                     linestyle=line_style, color=color, linewidth=2, alpha=0.8,
                     marker=marker, markersize=4, markevery=int(len(plot_time)/20))
        
        # Better styling for legend, title and labels
        plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        plt.title("Actual vs Predicted Energy Consumption", 
                  fontsize=14, fontweight='bold', pad=20)
        
        plt.xlabel("Time", fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel("Energy Consumption (kW)", fontsize=12, fontweight='bold', labelpad=10)
        
        # Improve the appearance of the x-axis
        plt.tick_params(axis='x', labelrotation=30)
        plt.tight_layout()
    
    # Convert plot to base64 image with higher quality
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')  # Increased DPI for better quality
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({"plot": plot_url})

@app.route('/api/models')
def get_models():
    """Returns a list of available models and their metrics"""
    if not models:
        return jsonify({"error": "No models available"})
    
    model_info = {}
    for name, data in models.items():
        model_info[name] = {
            'type': data['model_type'],
            'metrics': data['metrics']
        }
    
    return jsonify(model_info)

@app.route('/health')
def health_check():
    return jsonify({
        "status": "ok",
        "models_loaded": models_exist,
        "has_predictions": bool(predictions and len(predictions) > 0),
        "has_actual_values": bool(actual_values is not None),
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    app.run(debug=True) 