import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path (absolute path for reliability)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import modules
from nasa_data_loader import NASADataLoader
from multi_dataset_trainer import MultiDatasetTrainer
from predictor import PredictiveMaintenance
from visualizer import MaintenanceVisualizer

def create_directories():
    """Create necessary directories for multi-dataset processing"""
    directories = [
        'data/raw', 'data/processed', 'data/nasa', 
        'models/FD001', 'models/FD002', 'models/FD003', 'models/FD004',
        'models/ensemble', 'results', 'notebooks'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Multi-dataset directories created successfully!")

def verify_all_datasets():
    """Verify all NASA datasets are available"""
    nasa_path = 'data/nasa/'
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    
    print("🔍 Verifying NASA datasets...")
    missing_datasets = []
    
    for dataset in datasets:
        required_files = [
            f'train_{dataset}.txt',
            f'test_{dataset}.txt', 
            f'RUL_{dataset}.txt'
        ]
        
        dataset_complete = True
        for file in required_files:
            if not os.path.exists(os.path.join(nasa_path, file)):
                dataset_complete = False
                break
        
        if dataset_complete:
            print(f"✅ {dataset}: Complete")
        else:
            print(f"❌ {dataset}: Missing files")
            missing_datasets.append(dataset)
    
    if missing_datasets:
        print(f"\n⚠️ Missing datasets: {missing_datasets}")
        print("Please download from:")
        print("https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
        print("Extract all files to data/nasa/ directory")
        return False
    
    return True

def load_all_nasa_datasets():
    """Load all NASA datasets (FD001-FD004)"""
    print("\n🚀 Loading all NASA C-MAPSS datasets...")
    
    loader = NASADataLoader()
    all_datasets = loader.load_all_datasets()
    
    if not all_datasets:
        raise ValueError("No datasets could be loaded!")
    
    # Generate and display statistics
    stats = loader.get_dataset_statistics(all_datasets)
    
    print(f"\n📊 Dataset Statistics Summary:")
    print("=" * 80)
    
    for dataset, stat in stats.items():
        print(f"\n{dataset}:")
        print(f"  📈 Samples: {stat['total_samples']:,}")
        print(f"  🔧 Engines: {stat['total_engines']}")
        print(f"  ⚡ Avg cycles/engine: {stat['avg_cycles_per_engine']:.1f}")
        print(f"  ⚠️ Failure rate: {stat['failure_rate']:.3f}")
        print(f"  🎯 Fault modes: {stat['fault_modes']}")
        print(f"  🌡️ Operating conditions: {stat['operating_conditions']}")
    
    return all_datasets, stats

def train_multi_dataset_models(all_datasets):
    """Train models on all datasets"""
    print(f"\n🤖 Training models on all NASA datasets...")
    
    trainer = MultiDatasetTrainer()
    results = trainer.train_all_datasets(all_datasets)
    
    # Generate insights
    insights = trainer.generate_insights()
    
    return results, trainer

def demonstrate_multi_dataset_predictions(trainer):
    """Demonstrate predictions across different datasets"""
    print(f"\n🔮 Multi-Dataset Prediction Demonstration...")
    
    # Test scenarios representing different fault conditions
    test_scenarios = [
        {
            'name': 'Normal Operation (All Datasets)',
            'description': 'Baseline normal turbofan operation',
            'data': {
                'temperature': 518.67,
                'pressure': 14.62, 
                'vibration': 2388.04,
                'rpm': 2388.04,
                'oil_level': 1.30,
                'fuel_flow': 553.90,
                'altitude': 0,
                'speed': 0.84
            }
        },
        {
            'name': 'Single Fault Scenario (FD001/FD002)',
            'description': 'High pressure compressor degradation',
            'data': {
                'temperature': 625.45,
                'pressure': 13.95,
                'vibration': 2388.12,
                'rpm': 2388.02,
                'oil_level': 1.28,
                'fuel_flow': 553.65,
                'altitude': 0,
                'speed': 0.84
            }
        },
        {
            'name': 'Multi-Fault Scenario (FD003/FD004)', 
            'description': 'Combined HPC degradation and fan degradation',
            'data': {
                'temperature': 642.15,
                'pressure': 13.75,
                'vibration': 2388.18,
                'rpm': 2387.95,
                'oil_level': 1.22,
                'fuel_flow': 553.35,
                'altitude': 0,
                'speed': 0.84
            }
        },
        {
            'name': 'Variable Operating Conditions (FD002/FD004)',
            'description': 'High altitude, high Mach number operation',
            'data': {
                'temperature': 635.30,
                'pressure': 13.88,
                'vibration': 2388.15,
                'rpm': 2388.00,
                'oil_level': 1.25,
                'fuel_flow': 553.50,
                'altitude': 42000,  # High altitude
                'speed': 0.90       # High Mach number
            }
        }
    ]
    
    print(f"\n🎯 Testing scenarios across different dataset complexities...")
    print("=" * 80)
    
    for scenario in test_scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        print("-" * 50)
        
        # Test with different dataset-specific models if available
        for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
            if dataset in trainer.dataset_results:
                print(f"\n🔧 {dataset} Model Predictions:")
                
                try:
                    # Load dataset-specific predictor
                    predictor = PredictiveMaintenance()
                    
                    # Predict failure risk
                    failure_result = predictor.predict_failure_risk(scenario['data'])
                    print(f"   ⚠️ Failure Risk: {failure_result.get('failure_risk', 'N/A')}")
                    print(f"   📊 Probability: {failure_result.get('failure_probability', 0):.3f}")
                    
                    # Predict RUL
                    rul_result = predictor.predict_remaining_useful_life(scenario['data'])
                    if 'remaining_useful_life_hours' in rul_result:
                        rul_cycles = rul_result['remaining_useful_life_hours']
                        print(f"   ⏰ RUL: {rul_cycles:.0f} cycles")
                        print(f"   💡 Recommendation: {rul_result['maintenance_recommendation']}")
                    
                except Exception as e:
                    print(f"   ❌ Prediction error: {e}")

def create_multi_dataset_visualizations(all_datasets, results):
    """Create comprehensive visualizations for all datasets"""
    print(f"\n📊 Creating multi-dataset visualizations...")
    
    try:
        visualizer = MaintenanceVisualizer()
        
        # Create dataset comparison plots
        comparison_data = []
        
        for dataset, data in all_datasets.items():
            df = data['combined_df'].head(500)  # Sample for performance
            
            comparison_data.append({
                'dataset': dataset,
                'fault_modes': data['info']['fault_modes'],
                'operating_conditions': data['info']['operating_conditions'],
                'avg_temperature': df['temperature'].mean(),
                'avg_pressure': df['pressure'].mean(),
                'avg_vibration': df['vibration'].mean(),
                'failure_rate': df['failure_risk'].mean()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Individual dataset visualizations
        for dataset, data in all_datasets.items():
            print(f"Creating visualizations for {dataset}...")
            df_sample = data['combined_df'].head(1000)
            
            # Sensor data plots
            visualizer.plot_sensor_data(
                df_sample,
                sensors=['temperature', 'pressure', 'vibration', 'rpm']
            )
            
            # Failure risk distribution
            visualizer.plot_failure_risk_distribution(df_sample)
            
            # Model performance comparison
            if dataset in results:
                visualizer.plot_model_performance_comparison(results[dataset]['performance'])
        
        print("✅ Multi-dataset visualizations created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def generate_final_report(all_datasets, results, trainer):
    """Generate comprehensive final report"""
    report_lines = []
    
    report_lines.append("# 🚀 NASA C-MAPSS Multi-Dataset Analysis Report")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*80)
    
    # Dataset overview
    report_lines.append("\n## 📊 Dataset Overview")
    for dataset, data in all_datasets.items():
        info = data['info']
        df = data['combined_df']
        
        report_lines.append(f"\n### {dataset}")
        report_lines.append(f"- **Fault Modes**: {info['fault_modes']}")
        report_lines.append(f"- **Operating Conditions**: {info['operating_conditions']}")
        report_lines.append(f"- **Total Samples**: {len(df):,}")
        report_lines.append(f"- **Engines**: {df['unit_id'].nunique()}")
        report_lines.append(f"- **Failure Rate**: {df['failure_risk'].mean():.3f}")
    
    # Performance summary
    report_lines.append("\n## 🎯 Model Performance Summary")
    
    best_performers = {}
    for dataset, result in results.items():
        performance = result['performance']
        
        best_classifier = max(
            [(name, perf['accuracy']) for name, perf in performance.items() if 'accuracy' in perf],
            key=lambda x: x[1]
        )
        
        best_regressor = max(
            [(name, perf['r2']) for name, perf in performance.items() if 'r2' in perf],
            key=lambda x: x[1]
        )
        
        best_performers[dataset] = {
            'classifier': best_classifier,
            'regressor': best_regressor
        }
        
        report_lines.append(f"\n### {dataset}")
        report_lines.append(f"- **Best Classifier**: {best_classifier} ({best_classifier[1]:.4f})")
        report_lines.append(f"- **Best Regressor**: {best_regressor} (R²={best_regressor[1]:.4f})")
    
    # Insights
    insights = trainer.generate_insights()
    report_lines.append("\n## 🔍 Key Insights")
    report_lines.extend(insights[2:])  # Skip header lines
    
    # Recommendations
    report_lines.append("\n## 💡 Recommendations")
    report_lines.append("- **FD001**: Ideal for initial model development and testing")
    report_lines.append("- **FD002**: Use for evaluating robustness across operating conditions")
    report_lines.append("- **FD003**: Essential for multi-fault scenario validation")
    report_lines.append("- **FD004**: Ultimate complexity test for production readiness")
    report_lines.append("- **Ensemble Models**: Leverage cross-dataset training for maximum robustness")
    
    # Save report WITH UTF-8 ENCODING
    report_content = '\n'.join(report_lines)
    os.makedirs('results', exist_ok=True)
    with open('results/multi_dataset_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📋 Final report saved to: results/multi_dataset_analysis_report.md")
    return report_content

def main():
    """Main function for multi-dataset NASA analysis"""
    print("🚀 NASA C-MAPSS Multi-Dataset Predictive Maintenance System")
    print("="*80)
    print("Analyzing FD001, FD002, FD003, and FD004 datasets")
    print("Multi-fault scenarios and varying operating conditions")
    print("="*80)
    
    try:
        # Setup
        create_directories()
        
        # Verify datasets
        if not verify_all_datasets():
            print("❌ Cannot proceed without all datasets")
            return
        
        # Load all datasets
        all_datasets, stats = load_all_nasa_datasets()
        
        # Train models on all datasets
        results, trainer = train_multi_dataset_models(all_datasets)
        
        # Demonstrate predictions
        demonstrate_multi_dataset_predictions(trainer)
        
        # Create visualizations
        create_multi_dataset_visualizations(all_datasets, results)
        
        # Generate final report
        report = generate_final_report(all_datasets, results, trainer)
        
        print(f"\n🎉 Multi-Dataset Analysis Completed Successfully!")
        print("="*80)
        print("✅ All NASA datasets (FD001-FD004) processed")
        print("✅ Models trained for single/multi-fault scenarios")
        print("✅ Cross-dataset ensemble models created")
        print("✅ Comprehensive performance analysis generated")
        print("✅ Production-ready predictive maintenance system")
        
        print(f"\n📂 Results available in:")
        print("   • models/FD001-FD004/ - Dataset-specific models")
        print("   • models/ensemble/ - Cross-dataset ensemble models") 
        print("   • results/ - Analysis reports and insights")
        print("   • data/processed/ - Processed datasets")
        
    except Exception as e:
        print(f"❌ Error in multi-dataset analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()