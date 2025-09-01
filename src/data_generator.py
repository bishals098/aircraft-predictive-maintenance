# src/data_generator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import yaml

class SyntheticDataGenerator:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.sensors = self.config['sensors']
        self.n_samples = self.config['data']['synthetic_samples']
        
    def generate_normal_operation_data(self, n_samples):
        """Generate sensor data for normal aircraft operation"""
        data = {}
        
        # Base parameters for normal operation
        base_params = {
            'temperature': {'mean': 85, 'std': 10, 'range': (60, 120)},
            'pressure': {'mean': 14.7, 'std': 2, 'range': (10, 20)},
            'vibration': {'mean': 0.1, 'std': 0.05, 'range': (0, 0.5)},
            'rpm': {'mean': 3000, 'std': 300, 'range': (2000, 4000)},
            'oil_level': {'mean': 80, 'std': 10, 'range': (50, 100)},
            'fuel_flow': {'mean': 50, 'std': 8, 'range': (30, 80)},
            'altitude': {'mean': 35000, 'std': 5000, 'range': (0, 45000)},
            'speed': {'mean': 500, 'std': 50, 'range': (200, 600)}
        }
        
        for sensor in self.sensors:
            params = base_params[sensor]
            values = np.random.normal(params['mean'], params['std'], n_samples)
            values = np.clip(values, params['range'][0], params['range'][1])
            data[sensor] = values
        
        return data
    
    def generate_degradation_pattern(self, n_samples, failure_point=0.8):
        """Generate sensor data showing degradation pattern"""
        data = {}
        
        # Degradation patterns for each sensor
        degradation_factors = {
            'temperature': 1.5,  # Increases with wear
            'pressure': 0.8,     # Decreases with leaks
            'vibration': 3.0,    # Increases significantly
            'rpm': 0.9,          # Slightly decreases
            'oil_level': 0.7,    # Decreases with leaks
            'fuel_flow': 0.85,   # Decreases with blockages
            'altitude': 1.0,     # Remains constant (external factor)
            'speed': 0.95        # Slightly affected
        }
        
        base_data = self.generate_normal_operation_data(n_samples)
        
        # Create time-based degradation
        time_factor = np.linspace(1, failure_point, n_samples)
        
        for sensor in self.sensors:
            base_values = base_data[sensor]
            degradation = degradation_factors[sensor]
            
            # Apply gradual degradation
            if degradation > 1:  # Increasing trend
                degraded_values = base_values * (1 + (degradation - 1) * time_factor)
            else:  # Decreasing trend
                degraded_values = base_values * (degradation + (1 - degradation) * (1 - time_factor))
            
            # Add some noise
            noise = np.random.normal(0, base_values.std() * 0.1, n_samples)
            data[sensor] = degraded_values + noise
        
        return data
    
    def generate_complete_dataset(self):
        """Generate complete dataset with normal and failure patterns"""
        # Generate normal operation data (70%)
        normal_samples = int(self.n_samples * 0.7)
        normal_data = self.generate_normal_operation_data(normal_samples)
        normal_labels = np.zeros(normal_samples)  # 0 = Normal
        normal_rul = np.random.uniform(100, 200, normal_samples)  # Remaining Useful Life
        
        # Generate degradation data (30%)
        degradation_samples = self.n_samples - normal_samples
        degradation_data = self.generate_degradation_pattern(degradation_samples)
        degradation_labels = np.ones(degradation_samples)  # 1 = Failure Risk
        degradation_rul = np.linspace(50, 0, degradation_samples)  # Decreasing RUL
        
        # Combine datasets
        combined_data = {}
        for sensor in self.sensors:
            combined_data[sensor] = np.concatenate([
                normal_data[sensor], 
                degradation_data[sensor]
            ])
        
        # Create DataFrame
        df = pd.DataFrame(combined_data)
        df['failure_risk'] = np.concatenate([normal_labels, degradation_labels])
        df['remaining_useful_life'] = np.concatenate([normal_rul, degradation_rul])
        
        # Add timestamp
        start_date = datetime.now() - timedelta(days=len(df))
        df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq='H')
        
        # Add aircraft and component IDs
        df['aircraft_id'] = np.random.choice(['AC001', 'AC002', 'AC003'], len(df))
        df['component_id'] = np.random.choice(['ENG001', 'ENG002', 'HYD001'], len(df))
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df

    def save_synthetic_data(self, filepath='data/synthetic/aircraft_sensor_data.csv'):
        """Generate and save synthetic data"""
        df = self.generate_complete_dataset()
        df.to_csv(filepath, index=False)
        print(f"Synthetic data saved to {filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Failure risk distribution:\n{df['failure_risk'].value_counts()}")
        return df