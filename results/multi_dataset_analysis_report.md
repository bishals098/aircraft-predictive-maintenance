# NASA C-MAPSS Multi-Dataset Predictive Maintenance Analysis Report
Generated: 2025-09-17 20:56:44

## Dataset Overview

### FD001
- **Samples**: 33,727
- **Engines**: 100
- **Fault Modes**: 1
- **Operating Conditions**: 1

### FD002
- **Samples**: 87,750
- **Engines**: 260
- **Fault Modes**: 1
- **Operating Conditions**: 6

### FD003
- **Samples**: 41,316
- **Engines**: 100
- **Fault Modes**: 2
- **Operating Conditions**: 1

### FD004
- **Samples**: 102,463
- **Engines**: 249
- **Fault Modes**: 2
- **Operating Conditions**: 6

## Model Performance Summary

### Dataset Complexity vs Performance

| Dataset   |   Fault_Modes |   Operating_Conditions |   Best_Classifier_Accuracy |   Best_Regressor_R2 |   Complexity_Score |
|:----------|--------------:|-----------------------:|---------------------------:|--------------------:|-------------------:|
| FD001     |             1 |                      1 |                   0.9164   |            0.774731 |                  1 |
| FD003     |             2 |                      1 |                   0.893687 |            0.536309 |                  2 |
| FD002     |             1 |                      6 |                   0.903104 |            0.722459 |                  6 |
| FD004     |             2 |                      6 |                   0.878428 |            0.610971 |                 12 |

## Key Insights

- 
📊 Dataset Complexity Ranking (easiest to hardest):
-    FD001: 🟢 Simple (1 fault(s), 1 condition(s))
-    FD003: 🟡 Medium (2 fault(s), 1 condition(s))
-    FD002: 🔴 Complex (1 fault(s), 6 condition(s))
-    FD004: 🔴 Complex (2 fault(s), 6 condition(s))
- 
🏆 Best performing dataset: FD001 (0.9164 accuracy)
- ⚠️ Most challenging dataset: FD004 (0.8784 accuracy)
- 
🔧 Single-fault datasets average accuracy: 0.9098
- 🔧 Multi-fault datasets average accuracy: 0.8861
- 📉 Performance drop due to multiple faults: 0.0237
- 
🌡️ Single operating condition average accuracy: 0.9050
- 🌡️ Multiple operating conditions average accuracy: 0.8908
- 📉 Performance drop due to varying conditions: 0.0143

## Recommendations

1. **FD001 (Simple)**: Best for initial deployment and testing
2. **FD002 (Variable Conditions)**: Focus on environmental robustness
3. **FD003 (Multi-Fault)**: Enhance fault isolation capabilities
4. **FD004 (Complex)**: Requires advanced ensemble techniques

## Technical Specifications

- **Framework**: PyTorch + Scikit-learn ensemble
- **Models**: Random Forest, XGBoost, SVM, KNN, LSTM
- **Features**: 100+ engineered features per dataset
- **Deployment**: Real-time Streamlit dashboard