# src/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values and infinite values in the dataset"""
        print(f"Missing values before handling: {df.isnull().sum().sum()}")
        
        # Replace infinite values with NaN first
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill for time series data
        df = df.fillna(method='ffill')
        # Backward fill for any remaining NaN
        df = df.fillna(method='bfill')
        # Final fallback: fill remaining NaN with 0
        df = df.fillna(0)
        
        print(f"Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """Remove outliers using IQR method"""
        if columns is None:
            columns = self.config['sensors']
        
        original_shape = df.shape[0]
        
        for column in columns:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Skip if IQR is zero (constant values)
                if IQR == 0:
                    continue
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Remove outliers
                mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
                df = df[mask]
        
        removed_count = original_shape - df.shape[0]
        print(f"Outliers removed: {removed_count} rows ({removed_count/original_shape*100:.2f}%)")
        return df.reset_index(drop=True)
    
    def create_features(self, df):
        """Create additional features from sensor data with robust handling"""
        print("Creating engineered features...")
        
        sensor_cols = [col for col in self.config['sensors'] if col in df.columns]
        print(f"Working with sensors: {sensor_cols}")
        
        # Rolling statistics (with minimum periods to handle edge cases)
        for col in sensor_cols:
            df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
            df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
            df[f'{col}_rolling_max_5'] = df[col].rolling(window=5, min_periods=1).max()
            df[f'{col}_rolling_min_5'] = df[col].rolling(window=5, min_periods=1).min()
        
        # Lag features
        for col in sensor_cols:
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_5'] = df[col].shift(5)
        
        # Rate of change with robust handling
        for col in sensor_cols:
            df[f'{col}_rate_change'] = df[col].diff()
            
            # Safe percentage change calculation
            pct_change = df[col].pct_change()
            # Replace inf and -inf with 0, and limit extreme values
            pct_change = pct_change.replace([np.inf, -np.inf], 0)
            pct_change = np.clip(pct_change, -10, 10)  # Limit to ±1000%
            df[f'{col}_rate_change_pct'] = pct_change
        
        # Cross-sensor features with robust division
        if 'temperature' in df.columns and 'pressure' in df.columns:
            # Ensure pressure is never zero or too small
            pressure_safe = np.where(df['pressure'] < 1e-3, 1e-3, df['pressure'])
            temp_pressure_ratio = df['temperature'] / pressure_safe
            # Cap extreme ratios
            temp_pressure_ratio = np.clip(temp_pressure_ratio, 0, 1000)
            df['temp_pressure_ratio'] = temp_pressure_ratio
        else:
            df['temp_pressure_ratio'] = 0
        
        if 'vibration' in df.columns and 'rpm' in df.columns:
            # Ensure RPM is never zero or too small
            rpm_safe = np.where(df['rpm'] < 1, 1, df['rpm'])
            vibration_rpm_ratio = df['vibration'] / rpm_safe
            # Cap extreme ratios
            vibration_rpm_ratio = np.clip(vibration_rpm_ratio, 0, 10)
            df['vibration_rpm_ratio'] = vibration_rpm_ratio
        else:
            df['vibration_rpm_ratio'] = 0
        
        # Replace any remaining NaN, inf, or -inf values
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        # Final check: ensure all values are finite
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Replace any extreme values
            df[col] = np.where(np.abs(df[col]) > 1e10, 0, df[col])
            # Ensure all values are finite
            df[col] = np.where(np.isfinite(df[col]), df[col], 0)
        
        print(f"Feature engineering completed. Total features: {len(df.columns)}")
        return df
    
    def prepare_features_and_targets(self, df):
        """Prepare feature matrix and target variables"""
        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['timestamp', 'aircraft_id', 'component_id', 'failure_risk', 'remaining_useful_life', 'unit_id', 'time_cycles']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Selected {len(self.feature_columns)} features for training")
        
        # Save feature columns for prediction consistency
        os.makedirs('models', exist_ok=True)
        with open('models/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        X = df[self.feature_columns]
        y_classification = df['failure_risk'] if 'failure_risk' in df.columns else None
        y_regression = df['remaining_useful_life'] if 'remaining_useful_life' in df.columns else None
        
        print(f"Feature matrix shape: {X.shape}")
        if y_classification is not None:
            print(f"Classification targets: {y_classification.value_counts().to_dict()}")
        if y_regression is not None:
            print(f"Regression targets - mean: {y_regression.mean():.2f}, std: {y_regression.std():.2f}")
        
        return X, y_classification, y_regression
    
    def scale_features(self, X_train, X_test=None, fit_scaler=True):
        """Scale features using StandardScaler"""
        if fit_scaler:
            print("Fitting and transforming features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            # Save scaler
            joblib.dump(self.scaler, 'models/scaler.pkl')
            print("Scaler saved to models/scaler.pkl")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def create_sequences(self, X, y, sequence_length=50):
        """Create sequences for LSTM model"""
        # Convert to numpy arrays to use integer indexing
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X
        
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = y
        
        # Check if we have enough data for sequences
        if len(X_array) <= sequence_length:
            print(f"Warning: Not enough data for sequences. Data length: {len(X_array)}, Sequence length: {sequence_length}")
            return None, None
        
        X_seq, y_seq = [], []
        
        for i in range(len(X_array) - sequence_length + 1):
            X_seq.append(X_array[i:(i + sequence_length)])
            y_seq.append(y_array[i + sequence_length - 1])
        
        return np.array(X_seq), np.array(y_seq)
    
    def preprocess_data(self, df, create_sequences=False):
        """Complete preprocessing pipeline optimized for NASA data"""
        print("Starting NASA data preprocessing...")
        print(f"Initial NASA dataset shape: {df.shape}")
        
        # Check if this is NASA data format
        if 'unit_id' in df.columns:
            print("✅ NASA C-MAPSS dataset detected")
            print(f"Total engines: {df['unit_id'].nunique()}")
            print(f"Total operational cycles: {df['time_cycles'].sum()}")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers (more conservative for NASA data)
        df = self.remove_outliers(df, threshold=2.0)  # Less aggressive outlier removal
        
        # Reset index after outlier removal
        df = df.reset_index(drop=True)
        
        # Create features
        df = self.create_features(df)
        
        # Prepare features and targets
        X, y_class, y_reg = self.prepare_features_and_targets(df)
        
        # Check if we have valid data
        if X.empty:
            raise ValueError("No features available after preprocessing")
        
        # Split data while preserving engine groups for NASA data
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        print(f"Splitting NASA data with test_size={test_size}")
        
        if 'unit_id' in df.columns:
            # Split by engines to avoid data leakage
            unique_engines = df['unit_id'].unique()
            np.random.seed(random_state)
            np.random.shuffle(unique_engines)
            
            n_test_engines = int(len(unique_engines) * test_size)
            test_engines = unique_engines[:n_test_engines]
            train_engines = unique_engines[n_test_engines:]
            
            train_mask = df['unit_id'].isin(train_engines)
            test_mask = df['unit_id'].isin(test_engines)
            
            X_train = X[train_mask].reset_index(drop=True)
            X_test = X[test_mask].reset_index(drop=True)
            
            if y_class is not None:
                y_train_class = y_class[train_mask].reset_index(drop=True)
                y_test_class = y_class[test_mask].reset_index(drop=True)
            else:
                y_train_class = y_test_class = None
                
            if y_reg is not None:
                y_train_reg = y_reg[train_mask].reset_index(drop=True)
                y_test_reg = y_reg[test_mask].reset_index(drop=True)
            else:
                y_train_reg = y_test_reg = None
                
            print(f"NASA train engines: {len(train_engines)}")
            print(f"NASA test engines: {len(test_engines)}")
        else:
            # Standard split for non-NASA data
            if y_class is not None:
                # Check if we have enough samples for stratification
                class_counts = y_class.value_counts()
                if class_counts.min() < 2:
                    print("Warning: Not enough samples in some classes for stratification. Using random split.")
                    stratify = None
                else:
                    stratify = y_class
                
                X_train, X_test, y_train_class, y_test_class = train_test_split(
                    X, y_class, test_size=test_size, random_state=random_state, stratify=stratify
                )
            else:
                X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
                y_train_class = y_test_class = None
                
            if y_reg is not None:
                _, _, y_train_reg, y_test_reg = train_test_split(
                    X, y_reg, test_size=test_size, random_state=random_state
                )
            else:
                y_train_reg = y_test_reg = None
        
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Create sequences for NASA time series data
        if create_sequences:
            seq_length = self.config['models']['lstm']['sequence_length']
            print(f"Creating NASA time sequences with length {seq_length}...")
            
            if y_train_class is not None and len(X_train_scaled) > seq_length:
                X_train_seq, y_train_class_seq = self.create_sequences(X_train_scaled, y_train_class, seq_length)
                X_test_seq, y_test_class_seq = self.create_sequences(X_test_scaled, y_test_class, seq_length)
                
                if X_train_seq is not None:
                    print(f"NASA training sequences: {X_train_seq.shape}")
                    print(f"NASA test sequences: {X_test_seq.shape}")
                else:
                    print("Could not create sequences - insufficient data")
            else:
                X_train_seq = X_test_seq = None
                y_train_class_seq = y_test_class_seq = None
            
            return {
                'X_train': X_train_scaled, 'X_test': X_test_scaled,
                'X_train_seq': X_train_seq, 'X_test_seq': X_test_seq,
                'y_train_class': y_train_class, 'y_test_class': y_test_class,
                'y_train_class_seq': y_train_class_seq, 'y_test_class_seq': y_test_class_seq,
                'y_train_reg': y_train_reg, 'y_test_reg': y_test_reg,
                'feature_columns': self.feature_columns
            }
        
        return {
            'X_train': X_train_scaled, 'X_test': X_test_scaled,
            'y_train_class': y_train_class, 'y_test_class': y_test_class,
            'y_train_reg': y_train_reg, 'y_test_reg': y_test_reg,
            'feature_columns': self.feature_columns
        }