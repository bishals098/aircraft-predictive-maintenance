import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import joblib
import yaml
import warnings

warnings.filterwarnings('ignore')

class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence data"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMPredictor(nn.Module):
    """PyTorch LSTM model - equivalent to original TensorFlow Sequential"""
    def __init__(self, input_size, hidden_units, dropout=0.3):
        super(LSTMPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_units, hidden_units // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_units // 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, (hn, cn) = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Take last output
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze()

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
        """Train Random Forest regressor for RUL prediction with progress bar"""
        print("Training Random Forest Regressor...")
    
        from tqdm import tqdm
        import time
    
        rf_config = self.config['models']['random_forest']
        n_estimators = rf_config['n_estimators']
    
        # Create model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=rf_config['max_depth'],
            random_state=rf_config['random_state'],
            warm_start=True  # Enable incremental training
        )
    
        # Train with progress bar simulation
        print(f"Training {n_estimators} trees...")
    
        # For very small n_estimators, train all at once
        if n_estimators <= 10:
            with tqdm(total=100, desc="RF Regressor Progress", unit="%") as pbar:
                model.fit(X_train, y_train)
                pbar.update(100)
        else:
            # Train incrementally for progress display
            step_size = max(1, n_estimators // 20)  # Update progress 20 times
            current_trees = 0
        
            with tqdm(total=n_estimators, desc="RF Regressor Trees", unit="tree") as pbar:
                while current_trees < n_estimators:
                    # Calculate next step
                    next_trees = min(current_trees + step_size, n_estimators)
                    model.n_estimators = next_trees
                
                    # Fit with current number of trees
                    model.fit(X_train, y_train)
                
                    # Update progress bar
                    trees_added = next_trees - current_trees
                    pbar.update(trees_added)
                    current_trees = next_trees
                
                    # Small delay to make progress visible
                    time.sleep(0.1)
    
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

        # Auto-detect GPU for XGBoost
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tree_method = 'hist'  # Works for both CPU and GPU
        
        model = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            random_state=42,
            tree_method=tree_method,
            device=device
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
    
    def train_lstm_classifier(self, X_train_seq, y_train_seq, X_test_seq, y_test_seq):
        """Train PyTorch LSTM classifier"""
        print("Training LSTM Classifier...")
    
        # Add this import at the top of the file
        from tqdm import tqdm
    
        lstm_config = self.config['models']['lstm']
    
        # Get input dimensions
        input_size = X_train_seq.shape[2]
    
        # Initialize model
        model = LSTMPredictor(input_size, lstm_config['units'], lstm_config['dropout'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lstm_config.get('learning_rate', 0.001))
    
        # Create datasets and data loaders
        train_dataset = SequenceDataset(X_train_seq, y_train_seq)
        test_dataset = SequenceDataset(X_test_seq, y_test_seq)
    
        train_loader = DataLoader(train_dataset, batch_size=lstm_config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=lstm_config['batch_size'], shuffle=False)
    
        # Training loop with TensorFlow-style progress bars
        for epoch in range(lstm_config['epochs']):
            # Training phase with progress bar
            model.train()
            total_loss = 0
            correct = 0
            total = 0
        
            # RESTORED: TensorFlow-style progress bar for each epoch
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{lstm_config["epochs"]}')
        
            for batch_idx, (X_batch, y_batch) in enumerate(train_pbar):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
                # Update progress bar with current metrics (like TensorFlow)
                current_acc = correct / total if total > 0 else 0
                current_loss = total_loss / (batch_idx + 1)
                train_pbar.set_postfix({
                    'loss': f'{current_loss:.4f}', 
                    'acc': f'{current_acc:.4f}'
                })
        
            train_acc = correct / total
            avg_train_loss = total_loss / len(train_loader)
        
            # Validation phase (no progress bar needed for validation)
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
        
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                
                    predicted = (outputs > 0.5).float()
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
        
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(test_loader)
        
            # Print epoch summary (like TensorFlow does after progress bar)
            print(f" - val_loss: {avg_val_loss:.4f} - val_acc: {val_acc:.4f}")
    
        # Save model and continue with rest of function...
        torch.save(model.state_dict(), 'models/lstm_classifier.pth')
    
        # Final predictions
        y_pred_proba = []
        y_pred = []
    
        model.eval()
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                y_pred_proba.extend(outputs.cpu().numpy())
                y_pred.extend((outputs > 0.5).cpu().numpy().astype(int))
    
        # Evaluation
        accuracy = accuracy_score(y_test_seq, y_pred)
        report = classification_report(y_test_seq, y_pred, output_dict=True)
    
        self.models['lstm_classifier'] = model
        self.model_performance['lstm_classifier'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
        print(f"LSTM Classifier Accuracy: {accuracy:.4f}")
    
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