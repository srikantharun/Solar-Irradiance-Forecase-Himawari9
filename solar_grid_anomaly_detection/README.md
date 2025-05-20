# Solar Power Grid Sensor Anomaly Detection

This project implements an LSTM-based autoencoder for detecting anomalies in solar power grid sensor data. The system analyzes time-series data from solar grid sensors to identify unusual patterns that may indicate faults, failures, or other issues requiring attention.

## Project Structure

```
solar_grid_anomaly_detection/
├── data/                  # Data storage and processing scripts
├── models/                # LSTM autoencoder model implementation
├── notebooks/             # Jupyter notebooks for experiments and visualization
├── visualization/         # Visualization tools and utilities
└── utils/                 # Helper functions and utilities
```

## Features

- Time-series data preprocessing for solar grid sensors
- LSTM autoencoder for anomaly detection
- Visualization tools for anomaly identification
- Threshold-based anomaly scoring
- Performance evaluation metrics

## Usage

The main workflow involves:
1. Preparing sensor data
2. Training the LSTM autoencoder
3. Detecting anomalies using reconstruction error
4. Visualizing and analyzing results

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Inspiration

This project draws inspiration from the [sequitur](https://github.com/shobrook/sequitur) library, which provides autoencoders for sequential data.