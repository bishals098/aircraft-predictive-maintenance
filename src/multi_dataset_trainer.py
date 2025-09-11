import time
import numpy as np
import pandas as pd
import os
import joblib
import yaml
from datetime import datetime
from model_trainer import ModelTrainer
from data_preprocessor import DataPreprocessor

class MultiDatasetTrainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.dataset_results = {}
        self.ensemble_models = {}

    def train_all_datasets(self, all_datasets):
        """Train models on all NASA datasets"""
        print("🚀 Training models on all NASA datasets (FD001-FD004)...")
        
        for dataset, data in all_datasets.items():
            print(f"\n{'='*60}")
            print(f"Training models for NASA {dataset}")
            print(f"Fault modes: {data['info']['fault_modes']}")
            print(f"Operating conditions: {data['info']['operating_conditions']}")
            print(f"{'='*60}")
            
            # Preprocess data
            preprocessor = DataPreprocessor()
            data_dict = preprocessor.preprocess_data(
                data['combined_df'], 
                create_sequences=True
            )
            
            # Train models
            trainer = ModelTrainer()
            models, performance = trainer.train_all_models(data_dict)
            
            # Save dataset-specific models
            self.save_dataset_models(models, dataset)
            
            # Store results
            self.dataset_results[dataset] = {
                'models': models,
                'performance': performance,
                'data_dict': data_dict,
                'preprocessor': preprocessor,
                'dataset_info': data['info']
            }
            
            # Display performance
            self.display_dataset_performance(dataset, performance)
        
        # Create ensemble models across datasets
        self.create_cross_dataset_ensemble()
        
        return self.dataset_results

    def save_dataset_models(self, models, dataset):
        """Save models with dataset-specific naming"""
        dataset_model_dir = f'models/{dataset}'
        os.makedirs(dataset_model_dir, exist_ok=True)
        
        for model_name, model in models.items():
            if 'lstm' in model_name:
                model.save(f'{dataset_model_dir}/{model_name}.keras')
            else:
                joblib.dump(model, f'{dataset_model_dir}/{model_name}.pkl')
        
        print(f"✅ Models saved for {dataset}")

    def display_dataset_performance(self, dataset, performance):
        """Display performance metrics for each dataset"""
        print(f"\n📊 {dataset} Performance Summary:")
        print("-" * 50)
        
        classifiers = []
        regressors = []
        
        for model_name, metrics in performance.items():
            if 'accuracy' in metrics:
                classifiers.append((model_name, metrics['accuracy']))
            elif 'rmse' in metrics:
                regressors.append((model_name, metrics['rmse'], metrics['r2']))
        
        if classifiers:
            print("🎯 Classification Models:")
            for model_name, accuracy in sorted(classifiers, key=lambda x: x[1], reverse=True):
                print(f"   {model_name}: {accuracy:.4f}")
        
        if regressors:
            print("\n📈 Regression Models:")
            for model_name, rmse, r2 in sorted(regressors, key=lambda x: x[2], reverse=True):
                print(f"   {model_name}: RMSE={rmse:.4f}, R²={r2:.4f}")

    def create_cross_dataset_ensemble(self):
        """Create ensemble models trained across all datasets - OPTIMIZED VERSION"""
        print(f"\n🔄 Creating cross-dataset ensemble models...")
    
        # Combine all datasets for ensemble training
        combined_data = {
            'X_train': [],
            'y_train_class': [],
            'y_train_reg': []
        }
    
        total_samples = 0
        for dataset, results in self.dataset_results.items():
            data_dict = results['data_dict']
            combined_data['X_train'].append(data_dict['X_train'])
            combined_data['y_train_class'].append(data_dict['y_train_class'])
            combined_data['y_train_reg'].append(data_dict['y_train_reg'])
            total_samples += len(data_dict['X_train'])
    
        print(f"📊 Total combined samples: {total_samples:,}")
    
        # **OPTIMIZATION 1: Reduce sample size if too large**
        if total_samples > 50000:  # Limit to 50k samples max
            print(f"⚡ Large dataset detected ({total_samples:,} samples)")
            print("🔧 Applying sampling optimization for faster training...")
        
            # Sample from each dataset proportionally
            max_samples_per_dataset = 12500  # 50k / 4 datasets
            sampled_X, sampled_y_class, sampled_y_reg = [], [], []
        
            for i, (X, y_class, y_reg) in enumerate(zip(
                combined_data['X_train'], 
                combined_data['y_train_class'], 
                combined_data['y_train_reg']
            )):
                if len(X) > max_samples_per_dataset:
                    # Random sampling
                    indices = np.random.choice(len(X), max_samples_per_dataset, replace=False)
                    sampled_X.append(X[indices])
                    sampled_y_class.append(y_class.iloc[indices] if hasattr(y_class, 'iloc') else y_class[indices])
                    sampled_y_reg.append(y_reg.iloc[indices] if hasattr(y_reg, 'iloc') else y_reg[indices])
                else:
                    sampled_X.append(X)
                    sampled_y_class.append(y_class)
                    sampled_y_reg.append(y_reg)
        
            combined_data['X_train'] = sampled_X
            combined_data['y_train_class'] = sampled_y_class
            combined_data['y_train_reg'] = sampled_y_reg
        
            new_total = sum(len(X) for X in sampled_X)
            print(f"✅ Reduced dataset size: {new_total:,} samples ({(new_total/total_samples)*100:.1f}%)")
    
        # Stack all training data
        X_combined = np.vstack(combined_data['X_train'])
        y_class_combined = np.concatenate(combined_data['y_train_class'])
        y_reg_combined = np.concatenate(combined_data['y_train_reg'])
    
        print(f"🎯 Final training size: {X_combined.shape}")
    
        # **OPTIMIZATION 2: Use faster Random Forest parameters**
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
        # Optimized parameters for speed
        fast_params = {
            'n_estimators': 50,      # Reduced from 200
            'max_depth': 10,         # Limited depth
            'min_samples_split': 20, # Larger splits
            'min_samples_leaf': 10,  # Larger leaves
            'max_features': 'sqrt',  # Feature sampling
            'n_jobs': -1,            # Use all CPU cores
            'random_state': 42,
            'verbose': 1             # Show progress
        }
    
        print("🚀 Training ensemble classifier (optimized for speed)...")
        start_time = time.time()
    
        # Ensemble classifier with timeout monitoring
        ensemble_classifier = RandomForestClassifier(**fast_params)
    
        try:
            ensemble_classifier.fit(X_combined, y_class_combined)
            classifier_time = time.time() - start_time
            print(f"✅ Classifier trained in {classifier_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Classifier training failed: {e}")
            return
    
        print("🚀 Training ensemble regressor (optimized for speed)...")
        start_time = time.time()
    
        # Ensemble regressor
        ensemble_regressor = RandomForestRegressor(**fast_params)
    
        try:
            ensemble_regressor.fit(X_combined, y_reg_combined)
            regressor_time = time.time() - start_time
            print(f"✅ Regressor trained in {regressor_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Regressor training failed: {e}")
            return
    
        # Save ensemble models
        os.makedirs('models/ensemble', exist_ok=True)
        joblib.dump(ensemble_classifier, 'models/ensemble/cross_dataset_classifier.pkl')
        joblib.dump(ensemble_regressor, 'models/ensemble/cross_dataset_regressor.pkl')
    
        self.ensemble_models = {
            'classifier': ensemble_classifier,
           'regressor': ensemble_regressor
        }
    
        print("✅ Cross-dataset ensemble models created and saved")

    def compare_dataset_performance(self):
        """Compare performance across all datasets"""
        comparison_data = []
        
        for dataset, results in self.dataset_results.items():
            performance = results['performance']
            info = results['dataset_info']
            
            # Get best classifier and regressor performance
            best_classifier_acc = 0
            best_regressor_r2 = 0
            
            for model_name, metrics in performance.items():
                if 'accuracy' in metrics:
                    best_classifier_acc = max(best_classifier_acc, metrics['accuracy'])
                elif 'r2' in metrics:
                    best_regressor_r2 = max(best_regressor_r2, metrics['r2'])
            
            comparison_data.append({
                'Dataset': dataset,
                'Fault_Modes': info['fault_modes'],
                'Operating_Conditions': info['operating_conditions'],
                'Best_Classifier_Accuracy': best_classifier_acc,
                'Best_Regressor_R2': best_regressor_r2,
                'Complexity_Score': info['fault_modes'] * info['operating_conditions']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Complexity_Score')
        
        print(f"\n📋 Dataset Complexity vs Performance Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('models/dataset_performance_comparison.csv', index=False)
        
        return comparison_df

    def generate_insights(self):
        """Generate insights about multi-fault scenarios"""
        insights = []
        
        # Performance vs complexity analysis
        comparison_df = self.compare_dataset_performance()
        
        insights.append("🔍 NASA C-MAPSS Multi-Dataset Analysis Insights:")
        insights.append("=" * 60)
        
        # Dataset complexity ranking
        insights.append("\n📊 Dataset Complexity Ranking (easiest to hardest):")
        for _, row in comparison_df.iterrows():
            complexity = "🟢 Simple" if row['Complexity_Score'] == 1 else \
                        "🟡 Medium" if row['Complexity_Score'] <= 2 else "🔴 Complex"
            insights.append(f"   {row['Dataset']}: {complexity} "
                          f"({row['Fault_Modes']} fault(s), {row['Operating_Conditions']} condition(s))")
        
        # Performance insights
        best_dataset = comparison_df.loc[comparison_df['Best_Classifier_Accuracy'].idxmax()]
        worst_dataset = comparison_df.loc[comparison_df['Best_Classifier_Accuracy'].idxmin()]
        
        insights.append(f"\n🏆 Best performing dataset: {best_dataset['Dataset']} "
                       f"({best_dataset['Best_Classifier_Accuracy']:.4f} accuracy)")
        insights.append(f"⚠️ Most challenging dataset: {worst_dataset['Dataset']} "
                       f"({worst_dataset['Best_Classifier_Accuracy']:.4f} accuracy)")
        
        # Multi-fault vs single-fault comparison
        single_fault = comparison_df[comparison_df['Fault_Modes'] == 1]['Best_Classifier_Accuracy'].mean()
        multi_fault = comparison_df[comparison_df['Fault_Modes'] == 2]['Best_Classifier_Accuracy'].mean()
        
        insights.append(f"\n🔧 Single-fault datasets average accuracy: {single_fault:.4f}")
        insights.append(f"🔧 Multi-fault datasets average accuracy: {multi_fault:.4f}")
        insights.append(f"📉 Performance drop due to multiple faults: {(single_fault - multi_fault):.4f}")
        
        # Operating conditions impact
        single_cond = comparison_df[comparison_df['Operating_Conditions'] == 1]['Best_Classifier_Accuracy'].mean()
        multi_cond = comparison_df[comparison_df['Operating_Conditions'] == 6]['Best_Classifier_Accuracy'].mean()
        
        insights.append(f"\n🌡️ Single operating condition average accuracy: {single_cond:.4f}")
        insights.append(f"🌡️ Multiple operating conditions average accuracy: {multi_cond:.4f}")
        insights.append(f"📉 Performance drop due to varying conditions: {(single_cond - multi_cond):.4f}")
        
        # Save insights WITH UTF-8 ENCODING
        os.makedirs('models', exist_ok=True)
        with open('models/multi_dataset_insights.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(insights))
        
        # Print insights
        for insight in insights:
            print(insight)
        
        return insights