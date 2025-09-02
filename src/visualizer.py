# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class MaintenanceVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_sensor_data(self, df, sensors=None, figsize=(15, 10)):
        """Plot sensor data over time"""
        if sensors is None:
            sensors = ['temperature', 'pressure', 'vibration', 'rpm']
        
        fig, axes = plt.subplots(len(sensors), 1, figsize=figsize, sharex=True)
        if len(sensors) == 1:
            axes = [axes]
        
        for i, sensor in enumerate(sensors):
            if sensor in df.columns:
                axes[i].plot(df.index, df[sensor], color=self.colors[i % len(self.colors)])
                axes[i].set_ylabel(sensor.capitalize())
                axes[i].grid(True, alpha=0.3)
                
                # Highlight failure risk areas
                if 'failure_risk' in df.columns:
                    failure_mask = df['failure_risk'] == 1
                    if failure_mask.any():
                        axes[i].fill_between(df.index, df[sensor].min(), df[sensor].max(), 
                                           where=failure_mask, alpha=0.3, color='red', 
                                           label='Failure Risk')
        
        plt.xlabel('Time Index')
        plt.title('Aircraft Sensor Data Over Time')
        plt.tight_layout()
        plt.show()
    
    def plot_failure_risk_distribution(self, df):
        """Plot distribution of failure risk"""
        if 'failure_risk' in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Risk distribution pie chart
            plt.subplot(1, 2, 1)
            risk_counts = df['failure_risk'].value_counts()
            labels = ['Normal', 'Failure Risk']
            plt.pie(risk_counts.values, labels=labels, autopct='%1.1f%%', 
                   colors=['green', 'red'], startangle=90)
            plt.title('Failure Risk Distribution')
            
            # Risk over time
            plt.subplot(1, 2, 2)
            df['failure_risk'].rolling(window=100).mean().plot(color='red', linewidth=2)
            plt.title('Failure Risk Trend (Rolling Average)')
            plt.ylabel('Failure Risk Probability')
            plt.xlabel('Time Index')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def plot_model_performance_comparison(self, model_performance):
        """Compare performance of different models"""
        classifiers = {}
        regressors = {}
        
        for model_name, performance in model_performance.items():
            if 'accuracy' in performance:
                classifiers[model_name] = performance['accuracy']
            elif 'rmse' in performance:
                regressors[model_name] = {'rmse': performance['rmse'], 'r2': performance['r2']}
        
        if classifiers:
            plt.figure(figsize=(12, 8))
            
            # Classification models comparison
            plt.subplot(2, 2, 1)
            models = list(classifiers.keys())
            accuracies = list(classifiers.values())
            bars = plt.bar(models, accuracies, color=self.colors[:len(models)])
            plt.title('Model Accuracy Comparison')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
        
        if regressors:
            plt.subplot(2, 2, 2)
            reg_models = list(regressors.keys())
            rmse_values = [regressors[model]['rmse'] for model in reg_models]
            plt.bar(reg_models, rmse_values, color='orange')
            plt.title('Regressor RMSE Comparison')
            plt.ylabel('RMSE')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def create_interactive_dashboard(self, df, predictions=None):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Temperature Over Time', 'Pressure Over Time',
                          'Vibration Over Time', 'RPM Over Time',
                          'Failure Risk Probability', 'RUL Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sensor plots
        sensors = ['temperature', 'pressure', 'vibration', 'rpm']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for sensor, (row, col) in zip(sensors, positions):
            if sensor in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[sensor], name=sensor.capitalize(),
                             line=dict(color=self.colors[(row-1)*2 + col-1])),
                    row=row, col=col
                )
        
        # Failure risk plot
        if predictions and 'failure_probability' in predictions:
            fig.add_trace(
                go.Scatter(x=df.index, y=predictions['failure_probability'], 
                          name='Failure Risk', line=dict(color='red')),
                row=3, col=1
            )
        
        # RUL distribution
        if 'remaining_useful_life' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['remaining_useful_life'], name='RUL Distribution',
                           marker_color='blue', opacity=0.7),
                row=3, col=2
            )
        
        fig.update_layout(height=900, title_text="Aircraft Predictive Maintenance Dashboard")
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix for classification model"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Failure Risk'],
                   yticklabels=['Normal', 'Failure Risk'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Plot ROC curve for classification model"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_prediction_vs_actual(self, y_true, y_pred, model_name):
        """Plot prediction vs actual for regression models"""
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.title(f'Actual vs Predicted RUL - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted RUL')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # Distribution of residuals
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, color='purple')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()