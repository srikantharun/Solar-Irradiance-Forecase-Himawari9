"""
Synthetic data generator for solar power grid sensor data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_solar_sensor_data(
    num_days=30,
    num_sensors=5,
    sampling_interval_minutes=15,
    noise_level=0.05,
    anomaly_prob=0.02,
    anomaly_scale=3.0,
    seed=42
):
    """
    Generate synthetic time series data for solar power grid sensors.
    
    Args:
        num_days: Number of days to generate data for
        num_sensors: Number of sensors to simulate
        sampling_interval_minutes: Time between consecutive readings
        noise_level: Standard deviation of random noise
        anomaly_prob: Probability of an anomaly at each time step
        anomaly_scale: Scale factor for anomalies
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (data, anomaly_labels) where data is a DataFrame and
        anomaly_labels is a Series indicating anomalous points
    """
    np.random.seed(seed)
    
    # Calculate number of samples
    samples_per_day = int(24 * 60 / sampling_interval_minutes)
    total_samples = num_days * samples_per_day
    
    # Generate timestamps
    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(minutes=i*sampling_interval_minutes) 
                 for i in range(total_samples)]
    
    # Create DataFrame
    df = pd.DataFrame(index=timestamps)
    df.index.name = 'timestamp'
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # Initialize anomaly labels
    anomaly_labels = np.zeros(total_samples, dtype=int)
    
    # Generate data for each sensor
    for i in range(num_sensors):
        sensor_data = np.zeros(total_samples)
        
        for j in range(total_samples):
            timestamp = timestamps[j]
            hour = timestamp.hour
            
            # Base solar pattern (daylight dependent)
            if 6 <= hour < 20:  # Daylight hours
                # Solar peak around noon
                hour_factor = 1.0 - abs(hour - 13) / 7.0
                base_value = 0.8 * hour_factor
            else:  # Night
                base_value = 0.05  # Minimal production at night
            
            # Add some randomness
            sensor_data[j] = base_value + np.random.normal(0, noise_level)
            
            # Ensure non-negative values
            sensor_data[j] = max(0.01, sensor_data[j])
            
            # Randomly introduce anomalies
            if np.random.random() < anomaly_prob:
                if np.random.random() < 0.5:  # Spike anomaly
                    sensor_data[j] *= anomaly_scale
                else:  # Drop anomaly
                    sensor_data[j] /= anomaly_scale
                anomaly_labels[j] = 1
        
        # Add to DataFrame
        df[f'sensor_{i+1}'] = sensor_data
    
    # Add anomaly labels
    df['anomaly'] = anomaly_labels
    
    # Normalize sensor values to [0, 1] range
    sensor_cols = [f'sensor_{i+1}' for i in range(num_sensors)]
    for col in sensor_cols:
        df[col] = df[col] / df[col].max()
    
    return df


def create_sequences(df, target_cols, seq_length=24):
    """
    Create sequences for LSTM autoencoder.
    
    Args:
        df: DataFrame with time series data
        target_cols: List of column names to use as features
        seq_length: Sequence length
        
    Returns:
        Tuple of (X, y) where X contains sequences and y contains labels
    """
    X = []
    y = []
    
    for i in range(len(df) - seq_length):
        # Extract sequence
        sequence = df[target_cols].iloc[i:i+seq_length].values
        
        # Get label (1 if any point in sequence is anomalous, 0 otherwise)
        is_anomaly = 1 if df['anomaly'].iloc[i:i+seq_length].sum() > 0 else 0
        
        X.append(sequence)
        y.append(is_anomaly)
    
    return np.array(X), np.array(y)


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets, preserving temporal order.
    
    Args:
        X: Sequence data
        y: Labels
        test_size: Fraction of data to use for testing
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Determine split point
    split_idx = int(len(X) * (1 - test_size))
    
    # Split data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test