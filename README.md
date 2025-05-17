# Energy Consumption Prediction

This project predicts energy consumption using a Transformer-based deep learning model, with web visualization capabilities.

## Features

- Data preprocessing and feature selection
- Transformer model for time-series prediction
- LSTM model alternative implementation
- Web-based visualization dashboard
- Real-time model training and prediction

## Setup Instructions

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   
   Make sure the HomeC.csv file is in the root directory of the project.

3. **Running the Web Application**

   ```bash
   python app.py
   ```
   
   Then open your browser to http://localhost:5000

## Project Structure

- `energy_transformer.py`: Main implementation of the Transformer model
- `energy_visualization.py`: Alternative LSTM implementation
- `app.py`: Flask web application
- `templates/index.html`: Web dashboard template
- `HomeC.csv`: Energy consumption dataset

## Usage

1. Open the web interface
2. Click "Train Model" to start training the model
3. View performance metrics and visualizations

## Model Performance

The model evaluates performance using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

Lower MSE and MAE indicate better performance, while a higher R² Score (closer to 1.0) indicates a better fit. 