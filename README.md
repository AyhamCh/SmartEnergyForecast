# Smart Energy Forecast

A neural network-based project for predicting energy consumption based on environmental factors (temperature, vacuum, pressure, and humidity).

## Project Overview

This project implements a deep learning model to forecast energy consumption using environmental parameters. The model uses a multi-layer neural network trained on historical data to predict energy output with high accuracy.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Model Evaluation Metrics](#model-evaluation-metrics)

## Features

- Data cleaning and preprocessing
- Outlier detection using IQR method
- Feature scaling with StandardScaler
- Neural network with multiple hidden layers
- Comprehensive model evaluation
- Model persistence (save/load functionality)
- Visualization of training progress

## Dataset

The dataset (`energy.csv`) contains the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| Temperature | Ambient temperature | Celsius |
| Vacuum | Vacuum level | cm Hg |
| Pressure | Atmospheric pressure | millibar |
| Humidity | Relative humidity | % |
| Energy | Energy output (target variable) | MW |

**Dataset Statistics:**
- Total samples: 9,568 (after removing 41 duplicates)
- No missing values
- 80/20 train-test split

## Requirements

```
tensorflow>=2.x
pandas>=1.0.0
numpy>=1.17.0
scikit-learn>=0.24.0
matplotlib>=3.0.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-energy-forecast.git
cd smart-energy-forecast
```

2. Install required packages:
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

3. Ensure you have the dataset file `energy.csv` in the project directory.

## Project Structure

```
smart-energy-forecast/
│
├── SmartEnergyForecast.ipynb    # Main notebook
├── energy.csv                    # Dataset
├── energy_nn_model.h5           # Saved trained model
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Data Preprocessing

### 1. Data Cleaning

- **Duplicate Removal**: 41 duplicate rows removed
- **Missing Values**: No missing values detected
- **Data Types**: All features are float64

### 2. Outlier Detection

Using the Interquartile Range (IQR) method:

| Feature | Outliers Detected |
|---------|------------------|
| Temperature | 0 |
| Vacuum | 0 |
| Pressure | 91 |
| Humidity | 13 |
| Energy | 0 |

**Note**: Outliers were detected but not removed to preserve data integrity.

### 3. Feature Scaling

- Method: StandardScaler
- Applied to all input features
- Training set: (7,621, 4)
- Test set: (1,906, 4)

## Model Architecture

### Neural Network Structure

```python
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Params
=================================================================
Dense (Input Layer)         (None, 64)               320
Activation: ReLU

Dense (Hidden Layer)        (None, 32)               2,080
Activation: ReLU

Dense (Output Layer)        (None, 1)                33
=================================================================
Total params: 2,433
Trainable params: 2,433
Non-trainable params: 0
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Input Features | 4 (Temperature, Vacuum, Pressure, Humidity) |
| Hidden Layer 1 | 64 neurons, ReLU activation |
| Hidden Layer 2 | 32 neurons, ReLU activation |
| Output Layer | 1 neuron (Energy prediction) |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Metric | Mean Absolute Error (MAE) |

## Training

### Training Configuration

```python
Epochs: 30
Batch Size: 16
Validation Split: 20%
Optimizer: Adam (default learning rate)
```

### Training Progress

The model was trained for 30 epochs with the following final results:

- **Final Training Loss (MSE)**: 20.98
- **Final Training MAE**: 3.52
- **Final Validation Loss (MSE)**: 18.44
- **Final Validation MAE**: 3.39

## Results

### Model Performance on Test Set

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | 3.41 MW |
| Mean Squared Error (MSE) | 18.85 |
| Root Mean Squared Error (RMSE) | 4.34 MW |
| R² Score | 0.94 |
| Explained Variance Score | 0.94 |
| Mean Absolute Percentage Error (MAPE) | 0.75% |

### Key Insights

- The model achieves an excellent R² score of **0.94**, indicating that 94% of the variance in energy output is explained by the input features.
- The MAPE of **0.75%** demonstrates high prediction accuracy.
- The model converged well with minimal overfitting, as evidenced by similar training and validation losses.

## Usage

### Training the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load data
df = pd.read_csv('energy.csv')

# Prepare features and target
X = df[['Temperature', 'Vaccum', 'Pressure', 'Humidity']]
y = df['Energy']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=30, batch_size=16)
```

### Making Predictions

```python
from tensorflow import keras

# Load saved model
model = keras.models.load_model('energy_nn_model.h5')

# Prepare new data
new_data = [[20.5, 50.3, 1015.2, 65.8]]  # [Temperature, Vacuum, Pressure, Humidity]
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)
print(f"Predicted Energy: {prediction[0][0]:.2f} MW")
```

## Model Evaluation Metrics

### What Do These Metrics Mean?

**Mean Absolute Error (MAE)**: 3.41 MW
- On average, predictions are off by 3.41 MW from actual values
- Lower is better

**R² Score**: 0.94
- The model explains 94% of the variance in the data
- Range: 0 to 1, where 1 is perfect

**MAPE**: 0.75%
- Predictions are, on average, 0.75% away from actual values
- Excellent performance for practical applications

### Training Visualization

The loss curve shows steady convergence without significant overfitting, indicating a well-balanced model.

## Future Improvements

- Implement early stopping to prevent overfitting
- Experiment with different architectures (LSTM for time-series if temporal data is available)
- Add dropout layers for regularization
- Perform hyperparameter tuning (learning rate, batch size, neurons)
- Implement cross-validation for more robust evaluation
- Add feature engineering (polynomial features, interactions)





---

**Note**: This model is designed for educational and research purposes. For production deployment, additional validation and testing are recommended.
