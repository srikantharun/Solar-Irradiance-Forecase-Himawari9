# Solar Power Grid Anomaly Detection using LSTM Autoencoders

This project implements an anomaly detection system for solar power grid sensors using LSTM (Long Short-Term Memory) autoencoders. It can detect irregularities in solar power generation that may indicate sensor failures, shading issues, or other anomalies affecting energy production.

## Overview

Solar power generation data has many natural patterns, including:
- Daily cycles (sunrise to sunset)
- Seasonal variations (summer vs. winter)
- Weather effects (cloudy days, rain)
- Gradual panel degradation

However, anomalies can occur due to:
- Sensor malfunctions
- Partial panel shading
- Inverter issues
- Grid connection problems
- Dust/dirt accumulation

This project leverages deep learning to automatically detect these anomalies by learning the normal patterns in solar power generation data and identifying deviations.

## Features

- **Synthetic Data Generation**: Creates realistic solar power data with daily/seasonal patterns, weather effects, and injected anomalies
- **LSTM Autoencoder Model**: Deep learning model that learns to reconstruct normal sensor patterns
- **Anomaly Detection**: Identifies anomalies based on reconstruction error
- **Visualization Tools**: Comprehensive visualization utilities for data exploration and results analysis
- **Performance Evaluation**: Metrics and tools to assess anomaly detection performance

## Project Structure

```
LSTM/
├── data/                 # Data storage
├── models/               # Model definitions and saved models
│   ├── lstm_autoencoder.py   # LSTM autoencoder implementation
│   └── saved/            # Saved model weights
├── notebooks/            # Jupyter notebooks
│   └── solar_grid_anomaly_detection.ipynb  # Main workflow notebook
├── utils/                # Utility functions
│   ├── data_generator.py      # Synthetic data generation
│   └── visualization.py       # Visualization utilities
└── README.md             # Project documentation
```

## Installation and Dependencies

This project requires the following dependencies:

```
numpy
pandas
matplotlib
seaborn
torch
scikit-learn
```

You can install them using:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn
```

## Usage

### 1. Generate Synthetic Data

```python
from utils.data_generator import SolarPowerDataGenerator

# Initialize data generator
generator = SolarPowerDataGenerator(
    n_sensors=5,
    start_date="2023-01-01",
    end_date="2023-06-30",
    time_interval="15min",
    anomaly_percentage=0.02,
    random_seed=42
)

# Generate and save data
df_data, df_anomaly = generator.save_data(
    "data/solar_power_data.csv",
    "data/solar_power_anomalies.csv"
)
```

### 2. Create and Train the LSTM Autoencoder

```python
from models.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderTrainer

# Create the model
model = LSTMAutoencoder(
    input_dim=5,           # Number of sensors
    hidden_dim=64,
    latent_dim=32,
    sequence_length=96,    # 24 hours of 15-min data
    num_layers=2
)

# Create trainer
trainer = LSTMAutoencoderTrainer(
    model=model,
    sequence_length=96,
    batch_size=64,
    learning_rate=0.001
)

# Train the model
history = trainer.train(df_data, epochs=50, train_ratio=0.8)

# Save the model
trainer.save_model('models/saved/lstm_autoencoder.pt')
```

### 3. Detect Anomalies

```python
# Compute reconstruction errors and identify anomalies
errors, detected_anomalies, thresholds = trainer.detect_anomalies(df_data, threshold_percentile=99)
```

### 4. Visualize Results

```python
from utils.visualization import plot_solar_data, plot_reconstruction_error

# Plot the data with detected anomalies
plot_solar_data(df_data, anomalies=detected_anomalies)

# Plot reconstruction error
plot_reconstruction_error(errors, threshold=thresholds)
```

## Methodology

1. **Data Preprocessing**: The time series data is segmented into sequences and normalized.

2. **Model Architecture**: 
   - Encoder: LSTM layers that compress the input sequence into a latent representation
   - Decoder: LSTM layers that reconstruct the sequence from the latent representation

3. **Training**: The model is trained to minimize reconstruction error on normal data.

4. **Anomaly Detection**: Anomalies are identified by setting a threshold on the reconstruction error.

## Anomaly Types

The system can detect various types of anomalies:

- **Spike Anomalies**: Sudden increases in power output
- **Drop Anomalies**: Sudden decreases in power output
- **Drift Anomalies**: Gradual deviation from expected patterns
- **Stuck Values**: Sensor readings that remain constant when they should vary

## Performance Evaluation

The anomaly detection performance is evaluated using:

- Precision
- Recall
- F1 Score
- ROC and Precision-Recall curves
- Confusion matrix

## Extending the Project

To use this system with real solar power data:

1. Replace the synthetic data with your real sensor data
2. Adjust the sequence length based on your data's temporal resolution
3. Tune the model hyperparameters as needed
4. Experiment with different threshold values for anomaly detection

## Related Research

This project is inspired by research in time series anomaly detection using autoencoders:

- "Time Series Anomaly Detection Using LSTM Autoencoder with PyTorch in Python" (blog post)
- The sequitur library for anomaly detection in sequential data

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- This work is adapted from various research papers on time series anomaly detection
- The synthetic data generator is designed to simulate realistic solar power patterns