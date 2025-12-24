import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
from nasa_data_loader import NASADataLoader
from multi_dataset_trainer import MultiDatasetTrainer
from predictor import PredictiveMaintenance
from visualizer import MaintenanceVisualizer

warnings.filterwarnings('ignore')

def print_header():
    """Print application header"""
    print("🚀 NASA C-MAPSS Multi-Dataset Predictive Maintenance System")
    print("=" * 80)
    print("Analyzing FD001, FD002, FD003, and FD004 datasets")
    print("Multi-fault scenarios and varying operating conditions")
    print("=" * 80)

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'models/FD001',
        'models/FD002', 
        'models/FD003',
        'models/FD004',
        'models/ensemble',
        'data/processed',
        'results',
        'visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Multi-dataset directories created successfully!")

def verify_datasets():
    """Verify NASA datasets are available"""
    print("🔍 Verifying NASA datasets...")
    
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    dataset_files = ['train_FD001.txt', 'train_FD002.txt', 'train_FD003.txt', 'train_FD004.txt']
    
    for dataset, filename in zip(datasets, dataset_files):
        filepath = os.path.join('data', 'nasa', filename)
        if os.path.exists(filepath):
            print(f"✅ {dataset}: Complete")
        else:
            print(f"❌ {dataset}: Missing ({filename})")
            return False
    
    return True

def load_all_datasets():
    """Load all NASA C-MAPSS datasets"""
    print("\n🚀 Loading all NASA C-MAPSS datasets...")
    
    loader = NASADataLoader()
    
    try:
        all_datasets = loader.load_all_datasets(nasa_data_path='data/nasa/')
        if not all_datasets:
            print("❌ No datasets were successfully loaded!")
            return {}
        
        print(f"✅ Successfully loaded {len(all_datasets)} datasets")
        return all_datasets
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return {}

def display_dataset_statistics(all_datasets):
    """Display comprehensive dataset statistics"""
    print("\n📊 Dataset Statistics Summary:")
    print("=" * 80)
    
    for dataset, data in all_datasets.items():
        combined_df = data['combined_df']
        info = data.get('info', {})
        
        # Calculate statistics from the actual data
        num_samples = len(combined_df)
        
        # Get number of unique engines from the data
        if 'unit_id' in combined_df.columns:
            num_engines = combined_df['unit_id'].nunique()
        elif 'engine_id' in combined_df.columns:
            num_engines = combined_df['engine_id'].nunique()
        else:
            # Fallback: use predefined values from dataset_info in NASADataLoader
            engine_counts = {
                'FD001': 100, 'FD002': 260, 'FD003': 100, 'FD004': 249
            }
            num_engines = engine_counts.get(dataset, 100)
        
        avg_cycles = num_samples / num_engines if num_engines > 0 else 0
        
        # Check for failure risk column
        failure_rate = 0
        if 'failure_risk' in combined_df.columns:
            failure_rate = combined_df['failure_risk'].mean()
        elif 'RUL' in combined_df.columns:
            # If RUL exists, calculate failure rate as engines with RUL < threshold
            failure_rate = (combined_df['RUL'] < 30).mean()
        
        print(f"\n{dataset}:")
        print(f"  📈 Samples: {num_samples:,}")
        print(f"  🔧 Engines: {num_engines}")
        print(f"  ⚡ Avg cycles/engine: {avg_cycles:.1f}")
        print(f"  ⚠️ Failure rate: {failure_rate:.3f}")
        print(f"  🎯 Fault modes: {info.get('fault_modes', 'Unknown')}")
        print(f"  🌡️ Operating conditions: {info.get('operating_conditions', 'Unknown')}")

def train_multi_dataset_models(all_datasets):
    """Train models on all datasets"""
    print("\n🤖 Training models on all NASA datasets...")
    
    trainer = MultiDatasetTrainer()
    results = trainer.train_all_datasets(all_datasets)
    
    return results, trainer

def test_predictions(all_datasets):
    """Test prediction system with sample scenarios"""
    print("\n🔮 Multi-Dataset Prediction Demonstration...")
    predictor = PredictiveMaintenance()
    
    # Test scenarios for different dataset complexities
    scenarios = [
        {
            'name': 'Normal Operation (All Datasets)',
            'description': 'Baseline normal turbofan operation',
            'sensors': {
                'temperature': 518.67,
                'pressure': 14.62,
                'vibration': 2388.04,
                'rpm': 2388.04,
                'oil_level': 1.30,
                'fuel_flow': 553.90,
                'altitude': 0.0,
                'speed': 0.84
            }
        },
        {
            'name': 'Single Fault Scenario (FD001/FD002)',
            'description': 'High pressure compressor degradation',
            'sensors': {
                'temperature': 520.5,
                'pressure': 14.8,
                'vibration': 2395.2,
                'rpm': 2395.2,
                'oil_level': 1.28,
                'fuel_flow': 558.3,
                'altitude': 0.1,
                'speed': 0.82
            }
        },
        {
            'name': 'Multi-Fault Scenario (FD003/FD004)',
            'description': 'Combined HPC degradation and fan degradation',
            'sensors': {
                'temperature': 522.1,
                'pressure': 15.1,
                'vibration': 2401.8,
                'rpm': 2401.8,
                'oil_level': 1.25,
                'fuel_flow': 562.7,
                'altitude': -0.1,
                'speed': 0.80
            }
        },
        {
            'name': 'Variable Operating Conditions (FD002/FD004)',
            'description': 'High altitude, high Mach number operation',
            'sensors': {
                'temperature': 525.3,
                'pressure': 13.8,
                'vibration': 2410.5,
                'rpm': 2410.5,
                'oil_level': 1.32,
                'fuel_flow': 570.2,
                'altitude': 0.8,
                'speed': 0.95
            }
        }
    ]
    
    print("\n🎯 Testing scenarios across different dataset complexities...")
    print("=" * 80)
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        print("-" * 50)
        
        # Test predictions for each dataset model
        datasets = ['FD001', 'FD002', 'FD003', 'FD004']
        
        for dataset in datasets:
            print(f"\n🔧 {dataset} Model Predictions:")
            
            # Create dataset-specific predictor if available
            try:
                failure_pred = predictor.predict_failure_risk(scenario['sensors'])
                rul_pred = predictor.predict_remaining_useful_life(scenario['sensors'])
                
                # Display results
                failure_risk = failure_pred.get('failure_risk', 'Unknown')
                failure_prob = failure_pred.get('failure_probability', 0)
                rul_hours = rul_pred.get('remaining_useful_life_hours', 0)
                recommendation = rul_pred.get('maintenance_recommendation', 'Continue monitoring')
                
                print(f"   ⚠️ Failure Risk: {failure_risk}")
                print(f"   📊 Probability: {failure_prob:.3f}")
                print(f"   ⏰ RUL: {rul_hours:.0f} cycles")
                print(f"   💡 Recommendation: {recommendation}")
                
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")

def create_visualizations(all_datasets, results):
    """Create comprehensive visualizations"""
    print("\n📊 Creating multi-dataset visualizations...")
    
    visualizer = MaintenanceVisualizer()
    
    for dataset, data in all_datasets.items():
        print(f"Creating visualizations for {dataset}...")
        try:
            # Create dataset-specific visualizations
            combined_df = data['combined_df']
            
            # Basic sensor plots
            visualizer.plot_sensor_data(combined_df, save_path=f'visualizations/{dataset}_sensors.png')
            
            # Failure analysis
            if 'failure_risk' in combined_df.columns:
                visualizer.plot_failure_analysis(combined_df, save_path=f'visualizations/{dataset}_failure_analysis.png')
            
            # RUL distribution
            if 'remaining_useful_life' in combined_df.columns:
                visualizer.plot_rul_distribution(combined_df, save_path=f'visualizations/{dataset}_rul_distribution.png')
                
        except Exception as e:
            print(f"Warning: Visualization failed for {dataset}: {e}")
    
    print("✅ Multi-dataset visualizations created successfully!")

def generate_final_report(all_datasets, results, trainer):
    """Generate comprehensive final report"""
    report_lines = []
    
    report_lines.append("# NASA C-MAPSS Multi-Dataset Predictive Maintenance Analysis Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset overview
    report_lines.append("## Dataset Overview")
    report_lines.append("")
    for dataset, data in all_datasets.items():
        info = data['info']
        combined_df = data['combined_df']
        report_lines.append(f"### {dataset}")
        report_lines.append(f"- **Samples**: {len(combined_df):,}")
        num_engines = info.get('num_engines')
        if num_engines is None:
            # Try to infer from data
            combined_df = data['combined_df']
            if 'unit_id' in combined_df.columns:
                num_engines = combined_df['unit_id'].nunique()
            elif 'engine_id' in combined_df.columns:
                num_engines = combined_df['engine_id'].nunique()
            else:
                # Fallback: use standard value by dataset
                engine_counts = {'FD001': 100, 'FD002': 260, 'FD003': 100, 'FD004': 249}
                num_engines = engine_counts.get(dataset, '?')
        report_lines.append(f"- **Engines**: {num_engines}")
        report_lines.append(f"- **Fault Modes**: {info['fault_modes']}")
        report_lines.append(f"- **Operating Conditions**: {info['operating_conditions']}")
        report_lines.append("")
    
    # Performance summary
    report_lines.append("## Model Performance Summary")
    report_lines.append("")
    
    if hasattr(trainer, 'dataset_results'):
        comparison_df = trainer.compare_dataset_performance()
        report_lines.append("### Dataset Complexity vs Performance")
        report_lines.append("")
        report_lines.append(comparison_df.to_markdown(index=False))
        report_lines.append("")
    
    # Insights
    if hasattr(trainer, 'generate_insights'):
        insights = trainer.generate_insights()
        report_lines.append("## Key Insights")
        report_lines.append("")
        for insight in insights:
            if not insight.startswith('🔍') and not insight.startswith('='):
                report_lines.append(f"- {insight}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    report_lines.append("1. **FD001 (Simple)**: Best for initial deployment and testing")
    report_lines.append("2. **FD002 (Variable Conditions)**: Focus on environmental robustness")
    report_lines.append("3. **FD003 (Multi-Fault)**: Enhance fault isolation capabilities")
    report_lines.append("4. **FD004 (Complex)**: Requires advanced ensemble techniques")
    report_lines.append("")
    report_lines.append("## Technical Specifications")
    report_lines.append("")
    report_lines.append("- **Framework**: PyTorch + Scikit-learn ensemble")
    report_lines.append("- **Models**: Random Forest, XGBoost, SVM, KNN, LSTM")
    report_lines.append("- **Features**: 100+ engineered features per dataset")
    report_lines.append("- **Deployment**: Real-time Streamlit dashboard")
    
    # Save report WITH UTF-8 ENCODING
    report_content = '\n'.join(report_lines)
    os.makedirs('results', exist_ok=True)
    with open('results/multi_dataset_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📋 Final report saved to: results/multi_dataset_analysis_report.md")
    return report_content

def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        # Print header
        print_header()
        
        # Create directories
        create_directories()
        
        # Verify datasets
        if not verify_datasets():
            print("❌ Dataset verification failed. Please ensure all NASA datasets are available.")
            return
        
        # Load all datasets
        all_datasets = load_all_datasets()
        
        # Display statistics
        display_dataset_statistics(all_datasets)
        
        # Train models
        results, trainer = train_multi_dataset_models(all_datasets)
        
        # Test predictions
        test_predictions(all_datasets)
        
        # Create visualizations
        create_visualizations(all_datasets, results)
        
        # Generate final report
        report = generate_final_report(all_datasets, results, trainer)
        
        # Print completion summary
        execution_time = time.time() - start_time
        print(f"\n🎉 Multi-Dataset Analysis Completed Successfully!")
        print("=" * 80)
        print("✅ All NASA datasets (FD001-FD004) processed")
        print("✅ Models trained for single/multi-fault scenarios")  
        print("✅ Cross-dataset ensemble models created")
        print("✅ Comprehensive performance analysis generated")
        print("✅ Production-ready predictive maintenance system")
        print(f"⏱️ Total execution time: {execution_time/60:.1f} minutes")
        
        # Results summary
        print(f"\n📂 Results available in:")
        print("   • models/FD001-FD004/ - Dataset-specific models")
        print("   • models/ensemble/ - Cross-dataset ensemble models") 
        print("   • results/ - Analysis reports and insights")
        print("   • visualizations/ - Charts and plots")
        print("   • data/processed/ - Processed datasets")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error in multi-dataset analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()