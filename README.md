# aircraft-predictive-maintenance
Developed an aircraft turbofan engine predictive maintenance system using NASA's C-MAPSS dataset. Integrated ensemble machine learning and deep learning models to predict engine failure risk and estimate remaining useful life (RUL). Includes a real-time Streamlit dashboard for interactive monitoring and maintenance planning.
# 🚀 GitHub Repository Description for Your Aircraft Predictive Maintenance Project

***

# **Aircraft Predictive Maintenance System using NASA Turbofan Dataset**

A comprehensive predictive maintenance framework for aircraft turbofan engines utilizing NASA's C-MAPSS dataset. This system employs a diverse ensemble of machine learning and deep learning models—including Random Forest, XGBoost, SVM, KNN, and LSTM—to predict engine failure risks and estimate remaining useful life (RUL) across multiple fault and operational scenarios.

## ✨ **Features**

- **Multi-dataset training and validation** (FD001-FD004) covering complex fault conditions and operating environments
- **Robust feature engineering pipeline** addressing sensor noise, missing data, and outlier removal
- **Ensemble modeling techniques** combining classifier and regressor predictions for enhanced accuracy
- **Real-time interactive dashboard** built with Streamlit and Plotly for monitoring and predictions
- **Model explainability and uncertainty quantification** for safety-critical aerospace applications
- **Cross-dataset ensemble** training for improved generalization across fault scenarios

## 🛠️ **Technologies Used**

Scikit-learn | XGBoost | Streamlit | Plotly | pandas | numpy

## 🚀 **Quick Start**

### Prerequisites
- Python 3.12+
- Required packages (see `requirements.txt`)

### Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/aircraft-predictive-maintenance.git
cd aircraft-predictive-maintenance
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train all models**
```bash
python main_multi_dataset.py
```

4. **Launch the interactive dashboard**
```bash
streamlit run streamlit_app.py
```

## 📁 **Project Structure**

```
aircraft-predictive-maintenance/
├── main_multi_dataset.py          # Multi-dataset training pipeline
├── streamlit_app.py               # Interactive web dashboard
├── requirements.txt               # Python dependencies
├── src/                          # Core source modules
│   ├── data_preprocessor.py       # Data cleaning and feature engineering
│   ├── nasa_data_loader.py        # NASA C-MAPSS data loader
│   ├── model_trainer.py           # Individual model training
│   ├── multi_dataset_trainer.py   # Ensemble and cross-dataset training
│   ├── predictor.py               # Prediction engine
│   └── visualizer.py              # Data visualization
├── config/
│   └── config.yaml               # Configuration parameters
└── data/
    └── nasa/                     # NASA C-MAPSS dataset files
```

## 📊 **Performance Results**

| Dataset | Complexity | Best Model | Accuracy |
|---------|------------|------------|----------|
| **FD001** | Simple (1 fault, 1 condition) | Random Forest | **91.6%** |
| **FD002** | Medium (1 fault, 6 conditions) | XGBoost | **90.3%** |
| **FD003** | Medium (2 faults, 1 condition) | XGBoost | **89.4%** |
| **FD004** | Complex (2 faults, 6 conditions) | XGBoost | **87.8%** |

## 🎯 **Key Achievements**

- ✅ **High Accuracy**: Up to 91.6% failure prediction accuracy
- ✅ **Robust RUL Estimation**: Effective remaining useful life predictions
- ✅ **Multi-Fault Handling**: Successfully handles complex fault scenarios
- ✅ **Production Ready**: Complete end-to-end system with web interface
- ✅ **Scalable Architecture**: Modular design for easy extension

## 🔧 **Model Architecture**

The system implements a comprehensive ensemble approach:

- **Classification Models**: Random Forest, XGBoost, SVM, KNN, LSTM
- **Regression Models**: Random Forest Regressor for RUL estimation
- **Ensemble Strategy**: Voting classifier with optimized weights
- **Feature Engineering**: 100+ engineered features from 21 sensor parameters

## 📈 **Interactive Dashboard Features**

- Real-time sensor data visualization
- Failure risk assessment with confidence intervals
- Remaining useful life predictions
- Maintenance scheduling recommendations
- Historical trend analysis
- Multi-dataset model comparison

## 🧪 **Dataset Information**

This project uses NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset:

- **FD001**: Single fault mode, single operating condition
- **FD002**: Single fault mode, six operating conditions  
- **FD003**: Two fault modes, single operating condition
- **FD004**: Two fault modes, six operating conditions

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- NASA for providing the C-MAPSS turbofan engine dataset
- The open-source community for the excellent ML/DL libraries

***

**⭐ If you find this project useful, please consider giving it a star!**

***
