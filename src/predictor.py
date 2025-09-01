# src/predictor.py
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import yaml
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class PredictiveMaintenance:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.models['rf_classifier'] = joblib.load('models/random_forest_classifier.pkl')
            print("✓ Random Forest Classifier loaded")
        except:
            print("⚠ Random Forest Classifier not found")
            
        try:
            self.models['rf_regressor'] = joblib.load('models/random_forest_regressor.pkl')
            print("✓ Random Forest Regressor loaded")
        except:
            print("⚠ Random Forest Regressor not found")
            
        try:
            self.models['xgb_classifier'] = joblib.load('models/xgboost_classifier.pkl')
            print("✓ XGBoost Classifier loaded")
        except:
            print("⚠ XGBoost Classifier not found")
            
        try:
            self.models['knn_classifier'] = joblib.load('models/knn_classifier.pkl')
            print("✓ KNN Classifier loaded")
        except:
            print("⚠ KNN Classifier not found")
            
        try:
            self.models['svm_classifier'] = joblib.load('models/svm_classifier.pkl')
            print("✓ SVM Classifier loaded")
        except:
            print("⚠ SVM Classifier not found")
            
        try:
            self.scaler = joblib.load('models/scaler.pkl')
            print("✓ Scaler loaded")
        except:
            print("⚠ Scaler not found")
            
        try:
            with open('models/feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            print("✓ Feature columns loaded")
        except:
            print("⚠ Feature columns not found")
        
        try:
            self.models['lstm_classifier'] = tf.keras.models.load_model('models/lstm_classifier.keras')
            # Add compilation after loading
            self.models['lstm_classifier'].compile(
                optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy']
            )
            print("✓ LSTM Classifier loaded and compiled")
        except:
            print("⚠ LSTM Classifier not found")
    
    def preprocess_input(self, sensor_data):
        """Preprocess input sensor data for prediction"""
        if isinstance(sensor_data, dict):
            df = pd.DataFrame([sensor_data])
        elif isinstance(sensor_data, pd.DataFrame):
            df = sensor_data.copy()
        else:
            raise ValueError("Input must be dict or DataFrame")
        
        # Ensure all required sensor columns are present
        for sensor in self.config['sensors']:
            if sensor not in df.columns:
                df[sensor] = 0  # Default value for missing sensors
        
        # Create a dummy time series with multiple rows to enable proper feature engineering
        n_rows = 10  # Create 10 rows to enable rolling calculations
        sensor_cols = [col for col in self.config['sensors'] if col in df.columns]
        
        # Replicate the single row multiple times with slight variations
        extended_df = pd.DataFrame()
        for i in range(n_rows):
            row_data = {}
            for col in sensor_cols:
                base_value = df[col].iloc[0]
                # Add small random variation (±1%)
                variation = base_value * 0.01 * np.random.normal(0, 1)
                row_data[col] = base_value + variation
            extended_df = pd.concat([extended_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Apply same feature engineering as training
        # Rolling statistics
        for col in sensor_cols:
            extended_df[f'{col}_rolling_mean_5'] = extended_df[col].rolling(window=5, min_periods=1).mean()
            extended_df[f'{col}_rolling_std_5'] = extended_df[col].rolling(window=5, min_periods=1).std()
            extended_df[f'{col}_rolling_max_5'] = extended_df[col].rolling(window=5, min_periods=1).max()
            extended_df[f'{col}_rolling_min_5'] = extended_df[col].rolling(window=5, min_periods=1).min()
        
        # Lag features
        for col in sensor_cols:
            extended_df[f'{col}_lag_1'] = extended_df[col].shift(1)
            extended_df[f'{col}_lag_5'] = extended_df[col].shift(5)
        
        # Rate of change
        for col in sensor_cols:
            extended_df[f'{col}_rate_change'] = extended_df[col].diff()
            pct_change = extended_df[col].pct_change()
            pct_change = pct_change.replace([np.inf, -np.inf], 0)
            pct_change = np.clip(pct_change, -10, 10)
            extended_df[f'{col}_rate_change_pct'] = pct_change
        
        # Cross-sensor features
        if 'temperature' in extended_df.columns and 'pressure' in extended_df.columns:
            pressure_safe = np.where(extended_df['pressure'] < 1e-3, 1e-3, extended_df['pressure'])
            temp_pressure_ratio = extended_df['temperature'] / pressure_safe
            temp_pressure_ratio = np.clip(temp_pressure_ratio, 0, 1000)
            extended_df['temp_pressure_ratio'] = temp_pressure_ratio
        else:
            extended_df['temp_pressure_ratio'] = 0
        
        if 'vibration' in extended_df.columns and 'rpm' in extended_df.columns:
            rpm_safe = np.where(extended_df['rpm'] < 1, 1, extended_df['rpm'])
            vibration_rpm_ratio = extended_df['vibration'] / rpm_safe
            vibration_rpm_ratio = np.clip(vibration_rpm_ratio, 0, 10)
            extended_df['vibration_rpm_ratio'] = vibration_rpm_ratio
        else:
            extended_df['vibration_rpm_ratio'] = 0
        
        # Clean up any remaining issues
        extended_df = extended_df.replace([np.inf, -np.inf], 0)
        extended_df = extended_df.fillna(0)
        
        # Use saved feature columns if available
        if self.feature_columns is not None:
            # Ensure all required features are present
            for feature in self.feature_columns:
                if feature not in extended_df.columns:
                    extended_df[feature] = 0
            
            # Select only the features used during training, in the same order
            X = extended_df[self.feature_columns]
        else:
            # Fallback: create feature columns based on available data
            exclude_cols = ['timestamp', 'aircraft_id', 'component_id', 'failure_risk', 'remaining_useful_life']
            available_features = [col for col in extended_df.columns if col not in exclude_cols]
            X = extended_df[available_features]
        
        # Use only the last row (which represents the current sensor reading)
        X_final = X.iloc[[-1]]  # Take last row and keep as DataFrame
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_final)
        else:
            X_scaled = X_final.values
        
        return X_scaled, X_final.columns.tolist()
    
    def predict_failure_risk(self, sensor_data, model_name='ensemble'):
        """Predict failure risk using trained models"""
        try:
            X_scaled, feature_names = self.preprocess_input(sensor_data)
            
            predictions = {}
            
            # Individual model predictions
            for name, model in self.models.items():
                if 'classifier' in name and name != 'lstm_classifier':
                    try:
                        pred_proba = model.predict_proba(X_scaled)[:, 1]
                        predictions[name] = pred_proba[0] if len(pred_proba) == 1 else pred_proba
                    except Exception as e:
                        print(f"Warning: {name} prediction failed: {e}")
                        continue
            
            if model_name == 'ensemble':
                # Ensemble prediction (average of all models)
                if predictions:
                    ensemble_prob = np.mean(list(predictions.values()))
                    return {
                        'failure_probability': float(ensemble_prob),
                        'failure_risk': 'High' if ensemble_prob > self.config['alerts']['failure_threshold'] else 
                                       'Medium' if ensemble_prob > self.config['alerts']['warning_threshold'] else 'Low',
                        'individual_predictions': predictions
                    }
            elif model_name in predictions:
                prob = predictions[model_name]
                return {
                    'failure_probability': float(prob),
                    'failure_risk': 'High' if prob > self.config['alerts']['failure_threshold'] else 
                                   'Medium' if prob > self.config['alerts']['warning_threshold'] else 'Low'
                }
            
            # Fallback if no models available
            return {
                'failure_probability': 0.0,
                'failure_risk': 'Unknown',
                'error': 'No models available for prediction'
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def predict_remaining_useful_life(self, sensor_data):
        """Predict remaining useful life using regression model"""
        try:
            X_scaled, feature_names = self.preprocess_input(sensor_data)
            
            if 'rf_regressor' in self.models:
                try:
                    rul_pred = self.models['rf_regressor'].predict(X_scaled)
                    return {
                        'remaining_useful_life_hours': float(rul_pred[0]),
                        'remaining_useful_life_days': float(rul_pred[0] / 24),
                        'maintenance_recommendation': self.get_maintenance_recommendation(rul_pred[0])
                    }
                except Exception as e:
                    return {'error': f'RUL prediction failed: {e}'}
            
            return {'error': 'RUL model not available'}
            
        except Exception as e:
            return {'error': f'RUL prediction failed: {str(e)}'}
    
    def get_maintenance_recommendation(self, rul_hours):
        """Get maintenance recommendation based on RUL"""
        if rul_hours < 24:  # Less than 1 day
            return "URGENT: Schedule immediate maintenance"
        elif rul_hours < 168:  # Less than 1 week
            return "HIGH PRIORITY: Schedule maintenance within 24 hours"
        elif rul_hours < 720:  # Less than 1 month
            return "MEDIUM PRIORITY: Schedule maintenance within 1 week"
        else:
            return "LOW PRIORITY: Continue monitoring, schedule routine maintenance"
    
    def generate_maintenance_alerts(self, sensor_data, aircraft_id=None, component_id=None):
        """Generate comprehensive maintenance alerts"""
        failure_pred = self.predict_failure_risk(sensor_data)
        rul_pred = self.predict_remaining_useful_life(sensor_data)
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'aircraft_id': aircraft_id,
            'component_id': component_id,
            'failure_prediction': failure_pred,
            'rul_prediction': rul_pred,
            'alert_level': 'LOW',
            'actions_required': []
        }
        
        # Determine alert level and actions
        if failure_pred.get('failure_risk') == 'High':
            alert['alert_level'] = 'CRITICAL'
            alert['actions_required'].append('Immediate inspection required')
            alert['actions_required'].append('Ground aircraft if necessary')
        elif failure_pred.get('failure_risk') == 'Medium':
            alert['alert_level'] = 'WARNING'
            alert['actions_required'].append('Schedule enhanced monitoring')
            alert['actions_required'].append('Prepare maintenance resources')
        
        # Add RUL-based recommendations
        if rul_pred.get('remaining_useful_life_hours', float('inf')) < 168:  # Less than 1 week
            alert['actions_required'].append(rul_pred.get('maintenance_recommendation', ''))
        
        return alert
    
    def batch_predict(self, sensor_data_list):
        """Perform batch predictions on multiple sensor data points"""
        results = []
        
        for i, sensor_data in enumerate(sensor_data_list):
            try:
                failure_pred = self.predict_failure_risk(sensor_data)
                rul_pred = self.predict_remaining_useful_life(sensor_data)
                
                result = {
                    'index': i,
                    'failure_prediction': failure_pred,
                    'rul_prediction': rul_pred,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results