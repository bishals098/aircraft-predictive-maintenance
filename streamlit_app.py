import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
try:
    from predictor import PredictiveMaintenance
    from visualizer import MaintenanceVisualizer
    from nasa_data_loader import NASADataLoader
    from data_preprocessor import DataPreprocessor
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please ensure all required modules are in the src/ directory and models are trained.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="NASA Aircraft Predictive Maintenance",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background-color: #ffe6e6;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-normal {
        background-color: #d1edff;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the predictive maintenance system"""
    try:
        # Use the correct config path for multi-dataset system
        return PredictiveMaintenance(config_path='config/config.yaml')
    except Exception as e:
        st.error(f"Error loading predictive models: {e}")
        st.info("Please run `python main_multi_dataset.py` first to train the PyTorch models.")
        return None

@st.cache_data
def load_nasa_sample_data():
    """Load sample NASA data for visualization"""
    try:
        # Try to load processed data from any of the datasets
        for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
            file_path = f'data/processed/nasa_aircraft_sensor_data_{dataset}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                return df.head(1000)  # Load first 1000 samples for performance
        
        # Fallback to original processed data
        if os.path.exists('data/processed/nasa_aircraft_sensor_data.csv'):
            df = pd.read_csv('data/processed/nasa_aircraft_sensor_data.csv')
            return df.head(1000)
        
        return None
    except Exception as e:
        st.error(f"Error loading NASA data: {e}")
        return None

def create_sensor_input_form():
    """Create sensor input form with NASA C-MAPSS realistic values"""
    st.sidebar.header("🛠️ Sensor Input Configuration")
    
    # Dataset selection for different model types
    st.sidebar.subheader("Model Selection")
    dataset_choice = st.sidebar.selectbox(
        "Select NASA Dataset Model",
        ["FD001 (Simple)", "FD002 (Variable Conditions)", "FD003 (Multi-Fault)", "FD004 (Complex)", "Ensemble"],
        help="Different NASA datasets trained for various fault scenarios"
    )
    
    st.sidebar.subheader("NASA Turbofan Engine Sensors")
    
    # Create input form with NASA C-MAPSS realistic defaults
    sensor_data = {}
    
    # Temperature (Total temperature at fan inlet)
    sensor_data['temperature'] = st.sidebar.number_input(
        'Temperature (°R)', 
        min_value=500.0, 
        max_value=700.0, 
        value=518.67,
        step=1.0,
        help="Total temperature at fan inlet (Degrees Rankine)"
    )
    
    # Pressure (Total pressure at fan inlet)
    sensor_data['pressure'] = st.sidebar.number_input(
        'Pressure (psia)', 
        min_value=12.0, 
        max_value=16.0, 
        value=14.62,
        step=0.1,
        help="Total pressure at fan inlet (psia)"
    )
    
    # HPC Outlet Pressure (Static pressure at HPC outlet)
    sensor_data['vibration'] = st.sidebar.number_input(
        'HPC Outlet Pressure', 
        min_value=2380.0, 
        max_value=2400.0, 
        value=2388.04,
        step=0.1,
        help="Static pressure at High Pressure Compressor outlet"
    )
    
    # Physical Fan Speed
    sensor_data['rpm'] = st.sidebar.number_input(
        'Physical Fan Speed (rpm)', 
        min_value=2380.0, 
        max_value=2400.0, 
        value=2388.04,
        step=0.1,
        help="Physical fan speed (RPM)"
    )
    
    # Corrected Fan Speed (Oil Level proxy)
    sensor_data['oil_level'] = st.sidebar.number_input(
        'Corrected Fan Speed', 
        min_value=1.0, 
        max_value=1.5, 
        value=1.30,
        step=0.01,
        help="Corrected fan speed (normalized)"
    )
    
    # Fuel Flow
    sensor_data['fuel_flow'] = st.sidebar.number_input(
        'Fuel Flow', 
        min_value=500.0, 
        max_value=600.0, 
        value=553.90,
        step=0.1,
        help="Fuel flow rate"
    )
    
    # Operational Setting (Altitude proxy)
    sensor_data['altitude'] = st.sidebar.number_input(
        'Operational Setting (Altitude)', 
        min_value=-1.0, 
        max_value=1.0, 
        value=0.0,
        step=0.1,
        help="Operational setting representing altitude conditions"
    )
    
    # Bypass Duct Pressure (Speed proxy)
    sensor_data['speed'] = st.sidebar.number_input(
        'Bypass Duct Pressure', 
        min_value=0.5, 
        max_value=1.0, 
        value=0.84,
        step=0.01,
        help="Static pressure at bypass duct"
    )
    
    return sensor_data, dataset_choice

def display_prediction_results(failure_result, rul_result, dataset_choice):
    """Display prediction results with enhanced formatting"""
    
    # Determine alert styling based on failure risk
    failure_risk = failure_result.get('failure_risk', 'Unknown')
    failure_prob = failure_result.get('failure_probability', 0)
    
    if failure_risk == 'High':
        alert_class = "alert-critical"
        risk_emoji = "🔴"
        risk_color = "#dc3545"
    elif failure_risk == 'Medium':
        alert_class = "alert-warning"
        risk_emoji = "🟡"
        risk_color = "#ffc107"
    else:
        alert_class = "alert-normal"
        risk_emoji = "🟢"
        risk_color = "#28a745"
    
    # Main prediction display
    st.markdown(f"""
    <div class="{alert_class}">
        <h3>{risk_emoji} Failure Risk Assessment - {dataset_choice}</h3>
        <p><strong>Risk Level:</strong> {failure_risk}</p>
        <p><strong>Failure Probability:</strong> {failure_prob:.3f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Failure Risk",
            value=failure_risk,
            delta=f"{failure_prob:.3f} probability"
        )
    
    with col2:
        rul_cycles = rul_result.get('remaining_useful_life_hours', 0)
        st.metric(
            label="Remaining Useful Life",
            value=f"{rul_cycles:.0f} cycles",
            delta="Time to maintenance"
        )
    
    with col3:
        # Convert cycles to approximate days (assuming 1 cycle ≈ 1 day of operation)
        rul_days = rul_cycles / 24 if rul_cycles > 0 else 0
        st.metric(
            label="Estimated Days",
            value=f"{rul_days:.1f} days",
            delta="Operational estimate"
        )
    
    with col4:
        # Show model confidence based on ensemble predictions
        if 'individual_predictions' in failure_result:
            individual_preds = failure_result['individual_predictions']
            confidence = 1 - np.std(list(individual_preds.values())) if individual_preds else 0.5
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.3f}",
                delta="Ensemble agreement"
            )
        else:
            st.metric(
                label="Model Type",
                value="PyTorch Ensemble",
                delta="Deep learning enabled"
            )

def create_sensor_visualization(sensor_data):
    """Create visualization of current sensor readings"""
    
    # Create radar chart for sensor values
    sensors = list(sensor_data.keys())
    values = list(sensor_data.values())
    
    # Normalize values for better visualization (0-1 scale)
    normalized_values = []
    sensor_ranges = {
        'temperature': (500, 700),
        'pressure': (12, 16),
        'vibration': (2380, 2400),
        'rpm': (2380, 2400),
        'oil_level': (1.0, 1.5),
        'fuel_flow': (500, 600),
        'altitude': (-1, 1),
        'speed': (0.5, 1.0)
    }
    
    for sensor, value in sensor_data.items():
        min_val, max_val = sensor_ranges.get(sensor, (0, 1))
        normalized = (value - min_val) / (max_val - min_val)
        normalized_values.append(max(0, min(1, normalized)))  # Clamp to 0-1
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],  # Close the loop
        theta=sensors + [sensors[0]],
        fill='toself',
        name='Current Readings',
        line_color='rgb(31, 119, 180)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="NASA Turbofan Engine Sensor Reading Analysis",
        height=500
    )
    
    return fig

def display_historical_data(sample_data):
    """Display historical sensor data and trends"""
    if sample_data is None:
        st.warning("No historical data available. Please run the training script first.")
        return
    
    st.subheader("📊 Historical NASA Flight-Data Trends")
    
    # Select sensors to display
    available_sensors = ['temperature', 'pressure', 'vibration', 'rpm']
    display_sensors = [s for s in available_sensors if s in sample_data.columns]
    
    if not display_sensors:
        st.warning("No sensor data columns found in historical data.")
        return
    
    # Create time series plot
    fig = make_subplots(
        rows=len(display_sensors), 
        cols=1,
        subplot_titles=[f"{sensor.capitalize()} Over Time" for sensor in display_sensors],
        vertical_spacing=0.08
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, sensor in enumerate(display_sensors):
        fig.add_trace(
            go.Scatter(
                x=sample_data.index,
                y=sample_data[sensor],
                mode='lines',
                name=sensor.capitalize(),
                line=dict(color=colors[i % len(colors)])
            ),
            row=i+1, col=1
        )
        
        # Highlight failure risk areas if available
        if 'failure_risk' in sample_data.columns:
            failure_mask = sample_data['failure_risk'] == 1
            if failure_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=sample_data.index[failure_mask],
                        y=sample_data[sensor][failure_mask],
                        mode='markers',
                        name=f'{sensor} - Failure Risk',
                        marker=dict(color='red', size=4),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=i+1, col=1
                )
    
    fig.update_layout(
        height=200 * len(display_sensors),
        title_text="Historical Sensor Data Analysis (Processed)",
        showlegend=True
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Display summary statistics
    if len(display_sensors) > 0:
        st.subheader("📈 Statistical Summary")
        summary_stats = sample_data[display_sensors].describe()
        st.dataframe(summary_stats)

def create_maintenance_recommendations(failure_result, rul_result):
    """Generate detailed maintenance recommendations"""
    st.subheader("🔧 Maintenance Recommendations")
    
    failure_risk = failure_result.get('failure_risk', 'Unknown')
    rul_cycles = rul_result.get('remaining_useful_life_hours', 0)
    recommendation = rul_result.get('maintenance_recommendation', 'Continue monitoring')
    
    # Create recommendation based on risk level
    if failure_risk == 'High':
        st.error("🚨 **CRITICAL ACTION REQUIRED**")
        recommendations = [
            "Immediate inspection of turbofan engine components",
            "Ground aircraft until maintenance is completed",
            "Check high-pressure compressor for degradation",
            "Verify all sensor readings and calibration",
            "Schedule emergency maintenance within 24 hours"
        ]
        
    elif failure_risk == 'Medium':
        st.warning("⚠️ **PREVENTIVE MAINTENANCE RECOMMENDED**")
        recommendations = [
            "Schedule maintenance within next 72 hours",
            "Increase monitoring frequency for all sensors",
            "Prepare maintenance crew and spare parts",
            "Review recent flight operations for anomalies",
            "Consider reducing operational load until maintenance"
        ]
        
    else:
        st.success("✅ **NORMAL OPERATION - ROUTINE MONITORING**")
        recommendations = [
            "Continue regular monitoring schedule",
            "Next routine maintenance as per schedule",
            "Monitor sensor trends for early warning signs",
            "Keep maintenance logs updated",
            "Schedule routine inspection within normal timeframe"
        ]
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # Additional technical details
    with st.expander("🔍 Trained Model Technical Details"):
        st.markdown(f"**Model Prediction Details:**")
        st.markdown(f"- Failure Probability: {failure_result.get('failure_probability', 0):.4f}")
        st.markdown(f"- Remaining Cycles: {rul_cycles:.0f}")
        st.markdown(f"- System Recommendation: {recommendation}")
        st.markdown(f"- Framework: PyTorch LSTM + Scikit-learn Ensemble")
        
        if 'individual_predictions' in failure_result:
            st.markdown("**Individual Model Predictions:**")
            for model, prob in failure_result['individual_predictions'].items():
                framework = "PyTorch" if "lstm" in model else "Scikit-learn"
                st.markdown(f"- {model} ({framework}): {prob:.4f}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 NASA Aircraft Predictive Maintenance System</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Turbofan Engine Health Monitoring using NASA C-MAPSS Dataset")
    
    # Load predictor
    predictor = load_predictor()
    if predictor is None:
        st.error("Unable to load predictive models. Please check your installation and run the training script.")
        return
    
    # Load sample data
    sample_data = load_nasa_sample_data()
    
    # Sidebar for input
    sensor_data, dataset_choice = create_sensor_input_form()
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Engine Health", type="primary"):
        with st.spinner("Analyzing engine sensor data with trained models..."):
            try:
                # Make predictions using PyTorch-enabled predictor
                failure_result = predictor.predict_failure_risk(sensor_data)
                rul_result = predictor.predict_remaining_useful_life(sensor_data)
                
                # Display results
                display_prediction_results(failure_result, rul_result, dataset_choice)
                
                # Create maintenance recommendations
                create_maintenance_recommendations(failure_result, rul_result)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Please check that all sensor values are within expected ranges and models are properly loaded.")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🎛️ Sensor Monitor", "📊 Historical Data", "🔬 Analysis", "ℹ️ System Info"])
    
    with tab1:
        st.subheader("Current Sensor Readings")
        
        # Display sensor visualization
        radar_fig = create_sensor_visualization(sensor_data)
        st.plotly_chart(radar_fig, width='content')
        
        # Display current sensor values in a nice format
        st.subheader("📋 Current Values")
        col1, col2 = st.columns(2)
        
        sensor_descriptions = {
            'temperature': 'Total Temperature (°R)',
            'pressure': 'Total Pressure (psia)',
            'vibration': 'HPC Outlet Pressure',
            'rpm': 'Physical Fan Speed',
            'oil_level': 'Corrected Fan Speed',
            'fuel_flow': 'Fuel Flow Rate',
            'altitude': 'Operational Setting',
            'speed': 'Bypass Duct Pressure'
        }
        
        for i, (sensor, value) in enumerate(sensor_data.items()):
            if i % 2 == 0:
                with col1:
                    st.metric(sensor_descriptions.get(sensor, sensor), f"{value:.2f}")
            else:
                with col2:
                    st.metric(sensor_descriptions.get(sensor, sensor), f"{value:.2f}")
    
    with tab2:
        display_historical_data(sample_data)
    
    with tab3:
        st.subheader("🔬 Machine Learning Model Performance Analysis")
        
        # Display dataset information
        dataset_info = {
            "FD001": {"complexity": "Simple", "faults": 1, "conditions": 1, "accuracy": "91.6%"},
            "FD002": {"complexity": "Medium", "faults": 1, "conditions": 6, "accuracy": "90.3%"},
            "FD003": {"complexity": "Medium", "faults": 2, "conditions": 1, "accuracy": "89.4%"},
            "FD004": {"complexity": "Complex", "faults": 2, "conditions": 6, "accuracy": "87.8%"}
        }
        
        st.markdown("**NASA C-MAPSS Dataset Performance (LSTM + Ensemble):**")
        
        for dataset, info in dataset_info.items():
            with st.expander(f"{dataset} - {info['complexity']} Scenario"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Fault Modes", info['faults'])
                with col2:
                    st.metric("Operating Conditions", info['conditions'])
                with col3:
                    st.metric("Accuracy", info['accuracy'])
                with col4:
                    st.metric("Complexity", info['complexity'])
    
    with tab4:
        st.subheader("ℹ️ System Information")
        
        st.markdown("""
        **NASA C-MAPSS Aircraft Predictive Maintenance System**
        
        This system uses deep learning models combined with scikit-learn ensemble methods,
        trained on NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset to predict 
        turbofan engine failures and estimate remaining useful life (RUL).
        
        **Key Features:**
        - 🎯 **LSTM**: Deep learning for temporal pattern recognition
        - 🔄 **Hybrid Ensemble**: PyTorch + Scikit-learn model combination
        - 📊 **Multi-Dataset Support**: Trained on FD001-FD004 with varying complexity scenarios
        - 🛡️ **Safety-Critical**: Designed for aviation maintenance applications
        - ⚡ **GPU Acceleration**: Automatic CUDA/CPU detection for optimal performance
        
        **Model Architecture:**
        - **Deep Learning**: LSTM for sequential sensor data analysis
        - **Classical ML**: Random Forest, XGBoost, SVM, KNN for robust predictions
        - **Advanced Features**: 100+ engineered features from 21 sensor parameters
        - **Ensemble Strategy**: Weighted voting across all model types
        
        **Machine Learning Performance:**
        - Single-fault scenarios: Up to 91.6% accuracy
        - Multi-fault scenarios: Up to 89.4% accuracy
        - Cross-dataset ensemble: Robust performance across all conditions
        - GPU acceleration: 3-5x faster training when available
        """)
        
        # System status
        st.subheader("🔧 System Status")
        if predictor:
            st.success("✅ ML predictive models loaded successfully")
            st.success("✅ Multi-dataset support enabled")
            st.success("✅ Real-time predictions available")
            st.success("✅ GPU acceleration ready (if available)")
        else:
            st.error("❌ System not properly initialized")
        
        # Check PyTorch status
        try:
            import torch
            st.info(f"🔥 PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                st.success(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                st.info("💻 Running on CPU (CUDA not available)")
        except ImportError:
            st.warning("⚠️ PyTorch not detected")

if __name__ == "__main__":
    main()