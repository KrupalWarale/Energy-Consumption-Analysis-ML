# Energy Consumption Prediction

This web application for predicting household energy consumption using multiple machine learning models with interactive visualizations and real-time comparison capabilities.

## Overview

This project implements a sophisticated machine learning system that compares 6 different models (3 regression-based and 3 deep learning-based) to forecast household energy consumption based on environmental factors and historical usage patterns. The web interface provides interactive visualizations and metrics to help users understand model performance and prediction accuracy.

The system processes time-series data from smart meters combined with environmental factors (temperature, humidity, solar irradiance, etc.) to generate accurate energy consumption forecasts, which can help homeowners optimize their energy usage and reduce costs.

## Dataset

The project uses the HomeC.csv dataset, which contains comprehensive time-series data of household energy consumption along with various environmental factors collected at regular intervals.

### Dataset Details:
- **Size**: 124.9 MB
- **Time period**: Multiple months of household energy consumption
- **Sampling rate**: Regular intervals (typically minutes)
- **Features include**:
  - `time` - Timestamp of the measurement
  - `use [kW]` - Actual energy consumption in kilowatts (target variable)
  - `gen [kW]` - Energy generation (if applicable)
  - `temperature` - Ambient temperature in Celsius
  - `humidity` - Relative humidity percentage
  - `windSpeed` - Wind speed measurement
  - `Solar [kW]` - Solar irradiance measurement

- **Download Dataset**: [HomeC.csv (124.9 MB)]( <a href="https://drive.google.com/drive/u/2/folders/1SguX_Q28iCV0KrKY_rD6VyWrkt3eKXIM" target="_blank">HomeC.csv (124.9 MB)</a>)
- **Place the dataset** in the root directory of the project.

## Data Preprocessing

The system performs several preprocessing steps to prepare the data for modeling:

1. **Time Feature Extraction**: Converts timestamps to temporal features (hour, day of week, month, weekend flag)
2. **Feature Engineering**: Creates interaction features like temperature-humidity product and solar-wind combination
3. **Normalization**: Standardizes features using StandardScaler to ensure model stability
4. **Sequence Generation**: For deep learning models, creates time-sequence windows of features
5. **Train-Test Split**: Separates data into training and testing sets with appropriate temporal consideration
6. **Missing Value Handling**: Implements strategies to handle any gaps in the data

## Models Implemented

### Regression Models
1. **Random Forest Regressor**
   - Ensemble learning method using multiple decision trees (200 estimators)
   - Handles non-linear relationships effectively without feature scaling
   - Incorporates feature importance for energy consumption factors
   - Maximum depth of 15 nodes for detailed pattern capture
   - Implementation details: `n_estimators=200, max_depth=15, min_samples_split=5`
   
2. **Gradient Boosting Regressor**
   - Sequential ensemble method that builds trees to correct errors of previous trees
   - Good for complex datasets with various feature types
   - Learning rate of 0.1 with 200 estimators for balanced learning
   - Subsample ratio of 0.8 to prevent overfitting
   - Implementation details: `n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8`

3. **XGBoost Regressor**
   - Optimized implementation of gradient boosting with advanced regularization
   - High performance with gradient-based optimization and parallel processing
   - Utilizes both L1 and L2 regularization for model robustness
   - Column subsampling to prevent overfitting and improve generalization
   - Implementation details: `n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8`

### Deep Learning Models
1. **LSTM (Long Short-Term Memory)**
   - Recurrent neural network architecture with specialized memory cells
   - Designed for sequence modeling and time series forecasting
   - Capable of learning long-term dependencies in energy usage patterns
   - Multi-layered architecture with dropout for regularization
   - Implementation details: `hidden_dim=64, num_layers=2, dropout=0.2`

2. **GRU (Gated Recurrent Unit)**
   - Simplified version of LSTM with fewer parameters for faster training
   - Effective for capturing sequential patterns in energy consumption
   - Uses reset and update gates to control information flow
   - Balanced performance between computational efficiency and accuracy
   - Implementation details: `hidden_dim=64, num_layers=2, dropout=0.2`

3. **Hybrid CNN-LSTM**
   - Convolutional layers extract local features from time windows
   - LSTM layers model temporal dependencies across the sequence
   - Combines strengths of both architectures for improved forecasting
   - CNN component: `Conv1d layers (64, 128 filters)` with kernel size 3
   - LSTM component: `hidden_dim=64, num_layers=2`
   - Dense output layer for final prediction

## Model Training and Evaluation

The training process includes:

1. **Hyperparameter Optimization**: Fine-tuning model parameters for optimal performance
2. **Cross-Validation**: Ensuring model robustness across different time periods
3. **Early Stopping**: Preventing overfitting by monitoring validation performance
4. **Regularization Techniques**: Dropout, L1/L2 regularization based on model type
5. **Batch Processing**: Efficient training with appropriate batch sizes for deep learning models

Each model is thoroughly evaluated using:
- **R² Score**: Coefficient of determination (higher is better, best is 1.0)
- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **Training Time**: Computational efficiency measurement
- **Memory Usage**: Resource utilization monitoring

## Web Application Features

The Flask-based web application provides:

1. **Interactive Dashboard**: Comprehensive view of model performances and predictions
2. **Real-time Visualization**: Dynamic charts showing actual vs. predicted energy consumption
3. **Model Comparison**: Side-by-side metrics for all 6 models with color-coded performance indicators
4. **Time Series Analysis**: Interactive time series plots with zoom and pan capabilities
5. **Model Selection**: Ability to select specific models for focused analysis
6. **Responsive Design**: Mobile-friendly interface that adapts to different screen sizes
7. **Exportable Results**: Option to download prediction results and visualizations
8. **Error Analysis**: Visual representation of prediction errors across different time periods

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd energy-consumption-prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**
   - Download [HomeC.csv](https://drive.google.com/drive/u/2/folders/1SguX_Q28iCV0KrKY_rD6VyWrkt3eKXIM)
   - Place it in the project root directory

4. **Train Models**
   ```bash
   python train_and_save_model.py
   ```
   This script trains all 6 models and saves them in the 'models' directory. Depending on your hardware, this may take 30-60 minutes.

5. **Run the Web Application**
   ```bash
   python app.py
   ```
   Open your browser and navigate to http://localhost:5000

## Project Structure

- `app.py` - Flask web application with API endpoints and view functions
- `train_and_save_model.py` - Script to train and save all models with optimized hyperparameters
- `regression_models.py` - Implementation of regression models with feature engineering
- `deep_learning_models.py` - Implementation of deep learning architectures and training procedures
- `requirements.txt` - Required Python packages with version specifications
- `templates/` - HTML templates for the web interface using Bootstrap and Chart.js
- `models/` - Directory where trained models are saved in pickle format
- `static/` - CSS, JavaScript, and image files for the web interface

## Technical Details

### Data Processing Pipeline
```
Raw Data → Time Feature Extraction → Feature Engineering → Normalization → Model-specific Formatting → Train/Test Split
```

### Regression Model Pipeline
```
Feature Engineering → Scaling → Model Training → Hyperparameter Tuning → Evaluation → Model Persistence
```

### Deep Learning Pipeline
```
Sequence Generation → Tensor Conversion → Network Architecture → Batch Processing → Training Loop → Evaluation → Model Persistence
```

## Performance Considerations

- **Memory Usage**: The application is optimized to handle the large dataset through chunked processing
- **Computation Efficiency**: Models are trained with early stopping to reduce unnecessary computation
- **Scalability**: The design allows for easy addition of new models or features
- **Caching**: Predictions are cached to improve response time for the web interface

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 2.3+
- Pandas, NumPy, Matplotlib
- Scikit-learn
- XGBoost (optional but recommended)
- See requirements.txt for full list

## Browser Support

- Chrome
- Firefox
- Safari
- Edge

## Future Enhancements

- Integration with real-time energy monitoring systems
- Addition of transformer-based deep learning models
- Anomaly detection for unusual energy consumption patterns
- User authentication and personalized dashboards
- Mobile application for on-the-go monitoring



