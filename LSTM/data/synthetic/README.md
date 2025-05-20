# Synthetic Solar Power Grid Data

This directory contains synthetic solar power grid sensor data generated for anomaly detection experiments.

## Data Files

- `solar_power_data.csv`: Time series data with sensor readings
- `solar_power_anomalies.csv`: Anomaly labels for the data (1 = anomaly, 0 = normal)

## Data Generation

The data is generated using the `SolarPowerDataGenerator` class in `../../utils/data_generator.py`. It simulates realistic solar power production patterns with:

- Daily cycles (sun rises and sets)
- Seasonal variations (more power in summer, less in winter)
- Weather effects (cloudy days, random variations)
- Sensor degradation over time
- Random noise

## Anomaly Types

The generator injects several types of anomalies into the data:

1. **Spikes**: Sudden increases in power output (50-100% above normal)
2. **Drops**: Sudden decreases in power output (50-90% below normal)
3. **Drifts**: Gradual deviations that increase over several timestamps
4. **Stuck Values**: Sensors that report the same value for extended periods

## Data Format

The data files are CSV files with a datetime index and columns for each sensor.

### Example:

```
timestamp,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5
2023-01-01 00:00:00,0.0,0.0,0.0,0.0,0.0
2023-01-01 00:15:00,0.0,0.0,0.0,0.0,0.0
...
2023-01-01 12:00:00,320.5,412.7,276.1,390.2,350.8
...
```

## Using the Data

This synthetic data is designed for testing anomaly detection algorithms. You can use it to:

1. Train and evaluate LSTM autoencoder models
2. Test other anomaly detection techniques
3. Benchmark different algorithms against known anomalies
4. Visualize normal vs. anomalous patterns in solar power data

## Generating New Data

To generate new data with different characteristics, use the SolarPowerDataGenerator:

```python
from utils.data_generator import SolarPowerDataGenerator

# Initialize generator with custom parameters
generator = SolarPowerDataGenerator(
    n_sensors=5,               # Number of sensors
    start_date="2023-01-01",   # Start date
    end_date="2023-12-31",     # End date
    time_interval="15min",     # Time interval between readings
    anomaly_percentage=0.02,   # Percentage of anomalies
    random_seed=42             # For reproducibility
)

# Generate and save data
df_data, df_anomaly = generator.save_data(
    "solar_power_data.csv",
    "solar_power_anomalies.csv"
)
```

You can adjust parameters to create data with different characteristics, such as:
- More/fewer sensors
- Different time periods (seasons)
- Higher/lower anomaly rates
- Different time resolutions