import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SolarPowerDataGenerator:
    """
    Generator for synthetic solar power grid sensor data with anomalies
    Generates realistic solar power production patterns with diurnal cycles,
    weather effects, and seasonal variations
    """
    def __init__(
        self,
        n_sensors=5,
        start_date="2023-01-01",
        end_date="2023-12-31",
        time_interval="15min",
        anomaly_percentage=0.05,
        random_seed=42
    ):
        """
        Initialize the solar power data generator

        Parameters:
        -----------
        n_sensors : int
            Number of solar power sensors to simulate
        start_date : str
            Start date for the simulation in "YYYY-MM-DD" format
        end_date : str
            End date for the simulation in "YYYY-MM-DD" format
        time_interval : str
            Time interval between measurements (e.g., "15min", "1h")
        anomaly_percentage : float
            Percentage of data points to inject anomalies (0.0 to 1.0)
        random_seed : int
            Seed for reproducibility
        """
        self.n_sensors = n_sensors
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.anomaly_percentage = anomaly_percentage
        
        np.random.seed(random_seed)
        
        # Create date range
        self.date_range = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=time_interval
        )
        
        # Sensor base capacities (kW)
        self.sensor_capacities = np.random.uniform(
            low=200, 
            high=500, 
            size=n_sensors
        )
        
        # Sensor locations (affect weather patterns)
        self.sensor_weather_factors = np.random.uniform(
            low=0.8, 
            high=1.2, 
            size=n_sensors
        )
        
        # Sensor degradation over time (0.01% - 0.05% per day)
        self.sensor_degradation = np.random.uniform(
            low=0.0001, 
            high=0.0005, 
            size=n_sensors
        )
        
        # Generate anomaly timestamps
        self.anomaly_indices = self._generate_anomaly_indices()
        
    def _generate_base_production(self, timestamp):
        """
        Generate base production based on time of day and day of year
        
        Parameters:
        -----------
        timestamp : datetime
            Current timestamp
        
        Returns:
        --------
        float
            Base production factor (0.0 to 1.0)
        """
        # Extract hour and day of year
        hour = timestamp.hour + timestamp.minute / 60
        day_of_year = timestamp.dayofyear
        
        # No production during night hours (simplified)
        if hour < 6 or hour > 18:
            return 0.0
        
        # Calculate solar angle factor (peak at noon)
        solar_angle = np.sin(np.pi * (hour - 6) / 12)
        
        # Calculate seasonal factor (peak in summer, lowest in winter)
        if timestamp.year % 4 == 0 and (timestamp.year % 100 != 0 or timestamp.year % 400 == 0):
            days_in_year = 366
        else:
            days_in_year = 365
            
        season_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 172) / days_in_year)
        
        # Combine factors
        return max(0, solar_angle * season_factor)
    
    def _apply_weather_effects(self, base_production, timestamp, sensor_idx):
        """
        Apply random weather effects to base production
        
        Parameters:
        -----------
        base_production : float
            Base production factor
        timestamp : datetime
            Current timestamp
        sensor_idx : int
            Sensor index
        
        Returns:
        --------
        float
            Production after weather effects
        """
        # Get sensor weather factor
        weather_factor = self.sensor_weather_factors[sensor_idx]
        
        # Generate random cloud cover (higher probability in winter)
        month = timestamp.month
        winter_factor = 1.0 if month in [12, 1, 2] else 0.5
        
        # Random daily weather pattern (correlated throughout the day)
        day_seed = int(timestamp.strftime("%Y%m%d")) + sensor_idx
        np.random.seed(day_seed)
        daily_weather = np.random.uniform(0.7, 1.0)
        
        # Random clouds (short-term variations)
        np.random.seed(None)  # Reset seed
        cloud_factor = np.random.uniform(
            0.85, 
            1.0 if daily_weather > 0.9 else 0.95
        )
        
        # Apply weather effects
        return base_production * weather_factor * daily_weather * cloud_factor
    
    def _apply_degradation(self, production, timestamp, sensor_idx):
        """
        Apply sensor degradation over time
        
        Parameters:
        -----------
        production : float
            Current production
        timestamp : datetime
            Current timestamp
        sensor_idx : int
            Sensor index
        
        Returns:
        --------
        float
            Production after degradation
        """
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        days_since_start = (timestamp - start).days
        
        degradation_factor = 1.0 - (days_since_start * self.sensor_degradation[sensor_idx])
        
        return production * max(0.9, degradation_factor)  # Maximum 10% degradation
    
    def _generate_anomaly_indices(self):
        """
        Generate indices for anomaly injection
        
        Returns:
        --------
        list
            List of (timestamp_idx, sensor_idx) tuples for anomalies
        """
        total_timestamps = len(self.date_range)
        total_datapoints = total_timestamps * self.n_sensors
        
        # Calculate number of anomalies
        n_anomalies = int(total_datapoints * self.anomaly_percentage)
        
        # Generate random indices for anomalies
        anomaly_flat_indices = np.random.choice(
            total_datapoints, 
            size=n_anomalies, 
            replace=False
        )
        
        # Convert to (timestamp_idx, sensor_idx) format
        anomalies = []
        for flat_idx in anomaly_flat_indices:
            time_idx = flat_idx // self.n_sensors
            sensor_idx = flat_idx % self.n_sensors
            anomalies.append((time_idx, sensor_idx))
            
        return anomalies
    
    def _inject_anomalies(self, data):
        """
        Inject anomalies into the dataset
        
        Parameters:
        -----------
        data : numpy.ndarray
            Clean sensor data
        
        Returns:
        --------
        numpy.ndarray
            Data with injected anomalies
        tuple
            Anomaly labels (0 for normal, 1 for anomaly)
        """
        # Create a copy of the data
        anomalous_data = data.copy()
        
        # Create anomaly labels
        anomaly_labels = np.zeros((len(self.date_range), self.n_sensors), dtype=int)
        
        # Types of anomalies to inject
        anomaly_types = ['spike', 'drop', 'drift', 'stuck']
        
        for time_idx, sensor_idx in self.anomaly_indices:
            # Skip if the base value is zero (night time)
            if data[time_idx, sensor_idx] < 1.0:
                continue
                
            # Choose anomaly type
            anomaly_type = np.random.choice(anomaly_types)
            
            # Set the anomaly label
            anomaly_labels[time_idx, sensor_idx] = 1
            
            # Apply the anomaly
            if anomaly_type == 'spike':
                # Sudden spike in production (50-100% increase)
                anomalous_data[time_idx, sensor_idx] *= np.random.uniform(1.5, 2.0)
                
            elif anomaly_type == 'drop':
                # Sudden drop in production (50-90% decrease)
                anomalous_data[time_idx, sensor_idx] *= np.random.uniform(0.1, 0.5)
                
            elif anomaly_type == 'drift':
                # Gradual drift for several timestamps
                drift_length = np.random.randint(5, 20)
                drift_factor = np.random.uniform(1.3, 1.8)
                
                for i in range(drift_length):
                    if time_idx + i < len(self.date_range):
                        # Increase drift gradually
                        current_drift = 1.0 + (drift_factor - 1.0) * (i / drift_length)
                        anomalous_data[time_idx + i, sensor_idx] *= current_drift
                        anomaly_labels[time_idx + i, sensor_idx] = 1
                
            elif anomaly_type == 'stuck':
                # Sensor stuck at a value
                stuck_length = np.random.randint(10, 50)
                stuck_value = data[time_idx, sensor_idx]
                
                for i in range(stuck_length):
                    if time_idx + i < len(self.date_range):
                        anomalous_data[time_idx + i, sensor_idx] = stuck_value
                        anomaly_labels[time_idx + i, sensor_idx] = 1
        
        return anomalous_data, anomaly_labels
    
    def generate_data(self):
        """
        Generate complete synthetic solar power production dataset
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with timestamps and sensor readings
        pandas.DataFrame
            Anomaly labels (0 for normal, 1 for anomaly)
        """
        # Initialize data array
        data = np.zeros((len(self.date_range), self.n_sensors))
        
        # Generate data for each timestamp and sensor
        for time_idx, timestamp in enumerate(self.date_range):
            for sensor_idx in range(self.n_sensors):
                # Get base production factor
                base_production = self._generate_base_production(timestamp)
                
                # Apply sensor capacity
                production = base_production * self.sensor_capacities[sensor_idx]
                
                # Apply weather effects
                production = self._apply_weather_effects(production, timestamp, sensor_idx)
                
                # Apply degradation
                production = self._apply_degradation(production, timestamp, sensor_idx)
                
                # Add noise
                production += np.random.normal(0, 0.5)
                
                # Store value
                data[time_idx, sensor_idx] = max(0, production)
        
        # Inject anomalies
        anomalous_data, anomaly_labels = self._inject_anomalies(data)
        
        # Create DataFrames
        sensor_columns = [f'sensor_{i+1}' for i in range(self.n_sensors)]
        
        df_data = pd.DataFrame(
            anomalous_data, 
            index=self.date_range,
            columns=sensor_columns
        )
        
        df_anomaly = pd.DataFrame(
            anomaly_labels,
            index=self.date_range,
            columns=sensor_columns
        )
        
        return df_data, df_anomaly
    
    def save_data(self, data_path, anomaly_path=None):
        """
        Generate and save data to CSV files
        
        Parameters:
        -----------
        data_path : str
            Path to save the sensor data CSV
        anomaly_path : str, optional
            Path to save the anomaly labels CSV
        
        Returns:
        --------
        tuple
            DataFrames for sensor data and anomaly labels
        """
        # Generate data
        df_data, df_anomaly = self.generate_data()
        
        # Save sensor data
        df_data.to_csv(data_path)
        
        # Save anomaly labels if path provided
        if anomaly_path:
            df_anomaly.to_csv(anomaly_path)
            
        return df_data, df_anomaly


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = SolarPowerDataGenerator(
        n_sensors=5,
        start_date="2023-01-01",
        end_date="2023-03-31",  # 3 months of data
        time_interval="15min",
        anomaly_percentage=0.02,
        random_seed=42
    )
    
    # Generate and save data
    data, anomalies = generator.save_data(
        "solar_power_data.csv",
        "solar_power_anomalies.csv"
    )
    
    print(f"Generated {len(data)} timestamps for {data.shape[1]} sensors")
    print(f"Total anomalies: {anomalies.sum().sum()}")