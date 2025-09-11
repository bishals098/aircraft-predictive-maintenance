import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import joblib
import yaml
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {}
        self.model_performance = {}
    
    def train_random_forest_classifier(self, X_train, y_train, X_test, y_test):
        """Train Random Forest classifier for failure prediction"""
        print("Training Random Forest Classifier...")
        
        rf_config = self.config['models']['random_forest']
        model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            random_state=rf_config['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.models['random_forest_classifier'] = model
        self.model_performance['random_forest_classifier'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Random Forest Classifier Accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(model, 'models/random_forest_classifier.pkl')
        
        return model
    
    def train_random_forest_regressor(self, X_train, y_train, X_test, y_test):
        """Train Random Forest regressor for RUL prediction"""
        print("Training Random Forest Regressor...")
        
        rf_config = self.config['models']['random_forest']
        model = RandomForestRegressor(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            random_state=rf_config['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        self.models['random_forest_regressor'] = model
        self.model_performance['random_forest_regressor'] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"Random Forest Regressor RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Save model
        joblib.dump(model, 'models/random_forest_regressor.pkl')
        
        return model
    
    def train_xgboost_classifier(self, X_train, y_train, X_test, y_test):
        """Train XGBoost classifier"""
        print("Training XGBoost Classifier...")
        
        xgb_config = self.config['models']['xgboost']
        model = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.models['xgboost_classifier'] = model
        self.model_performance['xgboost_classifier'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"XGBoost Classifier Accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(model, 'models/xgboost_classifier.pkl')
        
        return model
    
    def train_svm_classifier(self, X_train, y_train, X_test, y_test):
        """Train SVM classifier"""
        print("Training SVM Classifier...")
        
        # Use a subset for SVM due to computational complexity
        subset_size = min(5000, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train_subset, y_train_subset)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.models['svm_classifier'] = model
        self.model_performance['svm_classifier'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"SVM Classifier Accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(model, 'models/svm_classifier.pkl')
        
        return model
    
    def train_knn_classifier(self, X_train, y_train, X_test, y_test):
        """Train K-Nearest Neighbors classifier"""
        print("Training KNN Classifier...")
        
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.models['knn_classifier'] = model
        self.model_performance['knn_classifier'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"KNN Classifier Accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(model, 'models/knn_classifier.pkl')
        
        return model
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        lstm_config = self.config['models']['lstm']
        
        model = Sequential([
            LSTM(lstm_config['units'], return_sequences=True, input_shape=input_shape),
            Dropout(lstm_config['dropout']),
            LSTM(lstm_config['units'] // 2, return_sequences=False),
            Dropout(lstm_config['dropout']),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train_lstm_classifier(self, X_train_seq, y_train_seq, X_test_seq, y_test_seq):
        """Train LSTM classifier for sequence data"""
        print("Training LSTM Classifier...")
        
        lstm_config = self.config['models']['lstm']
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = self.build_lstm_model(input_shape)
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=lstm_config['epochs'],
            batch_size=lstm_config['batch_size'],
            validation_data=(X_test_seq, y_test_seq),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test_seq).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Evaluation
        accuracy = accuracy_score(y_test_seq, y_pred)
        report = classification_report(y_test_seq, y_pred, output_dict=True)
        
        self.models['lstm_classifier'] = model
        self.model_performance['lstm_classifier'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'history': history
        }
        
        print(f"LSTM Classifier Accuracy: {accuracy:.4f}")
        
        # Save model
        model.save('models/lstm_classifier.keras')
        
        return model
    
    def train_all_models(self, data_dict):
        """Train all models with provided data"""
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train_class = data_dict['y_train_class']
        y_test_class = data_dict['y_test_class']
        y_train_reg = data_dict['y_train_reg']
        y_test_reg = data_dict['y_test_reg']
        
        # Classification models
        if y_train_class is not None:
            self.train_random_forest_classifier(X_train, y_train_class, X_test, y_test_class)
            self.train_xgboost_classifier(X_train, y_train_class, X_test, y_test_class)
            self.train_knn_classifier(X_train, y_train_class, X_test, y_test_class)
            self.train_svm_classifier(X_train, y_train_class, X_test, y_test_class)
        
        # Regression models
        if y_train_reg is not None:
            self.train_random_forest_regressor(X_train, y_train_reg, X_test, y_test_reg)
        
        # LSTM model (if sequence data is available)
        if 'X_train_seq' in data_dict and data_dict['X_train_seq'] is not None:
            self.train_lstm_classifier(
                data_dict['X_train_seq'], data_dict['y_train_class_seq'],
                data_dict['X_test_seq'], data_dict['y_test_class_seq']
            )
        
        return self.models, self.model_performance
    
    def get_model_comparison(self):
        """Get comparison of all trained models"""
        comparison = {}
        
        for model_name, performance in self.model_performance.items():
            if 'accuracy' in performance:
                comparison[model_name] = {
                    'accuracy': performance['accuracy'],
                    'type': 'classifier'
                }
            elif 'rmse' in performance:
                comparison[model_name] = {
                    'rmse': performance['rmse'],
                    'r2': performance['r2'],
                    'type': 'regressor'
                }
        
        return comparison
#bishals098