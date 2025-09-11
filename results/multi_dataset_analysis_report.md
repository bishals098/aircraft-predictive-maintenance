# 🚀 NASA C-MAPSS Multi-Dataset Analysis Report
Generated on: 2025-09-11 19:10:47

================================================================================

## 📊 Dataset Overview

### FD001
- **Fault Modes**: 1
- **Operating Conditions**: 1
- **Total Samples**: 33,727
- **Engines**: 100
- **Failure Rate**: 0.355

### FD002
- **Fault Modes**: 1
- **Operating Conditions**: 6
- **Total Samples**: 87,750
- **Engines**: 260
- **Failure Rate**: 0.341

### FD003
- **Fault Modes**: 2
- **Operating Conditions**: 1
- **Total Samples**: 41,316
- **Engines**: 100
- **Failure Rate**: 0.371

### FD004
- **Fault Modes**: 2
- **Operating Conditions**: 6
- **Total Samples**: 102,463
- **Engines**: 249
- **Failure Rate**: 0.326

## 🎯 Model Performance Summary

### FD001
- **Best Classifier**: ('random_forest_classifier', 0.9163996948893974) (0.9164)
- **Best Regressor**: ('random_forest_regressor', 0.7747313868482438) (R²=0.7747)

### FD002
- **Best Classifier**: ('xgboost_classifier', 0.903104354621013) (0.9031)
- **Best Regressor**: ('random_forest_regressor', 0.722458585939107) (R²=0.7225)

### FD003
- **Best Classifier**: ('xgboost_classifier', 0.893687230989957) (0.8937)
- **Best Regressor**: ('random_forest_regressor', 0.5363085279874035) (R²=0.5363)

### FD004
- **Best Classifier**: ('xgboost_classifier', 0.878427566124727) (0.8784)
- **Best Regressor**: ('random_forest_regressor', 0.6109705788033197) (R²=0.6110)

## 🔍 Key Insights

📊 Dataset Complexity Ranking (easiest to hardest):
   FD001: 🟢 Simple (1 fault(s), 1 condition(s))
   FD003: 🟡 Medium (2 fault(s), 1 condition(s))
   FD002: 🔴 Complex (1 fault(s), 6 condition(s))
   FD004: 🔴 Complex (2 fault(s), 6 condition(s))

🏆 Best performing dataset: FD001 (0.9164 accuracy)
⚠️ Most challenging dataset: FD004 (0.8784 accuracy)

🔧 Single-fault datasets average accuracy: 0.9098
🔧 Multi-fault datasets average accuracy: 0.8861
📉 Performance drop due to multiple faults: 0.0237

🌡️ Single operating condition average accuracy: 0.9050
🌡️ Multiple operating conditions average accuracy: 0.8908
📉 Performance drop due to varying conditions: 0.0143

## 💡 Recommendations
- **FD001**: Ideal for initial model development and testing
- **FD002**: Use for evaluating robustness across operating conditions
- **FD003**: Essential for multi-fault scenario validation
- **FD004**: Ultimate complexity test for production readiness
- **Ensemble Models**: Leverage cross-dataset training for maximum robustness