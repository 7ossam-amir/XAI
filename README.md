# 🩺 Sepsis Prediction Using Machine Learning

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![GitHub stars](https://img.shields.io/github/stars/your-username/sepsis-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/sepsis-prediction?style=social)

A machine learning project to predict sepsis in patients using vital signs and lab results. This repository implements a Random Forest model to identify septic cases with high accuracy, leveraging features like FiO2, PaCO2, Bilirubin, and Lactate.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🌟 Project Overview
Sepsis is a life-threatening condition caused by the body's response to infection. Early detection can save lives. This project uses a Random Forest model to predict sepsis by analyzing clinical features such as FiO2, PaCO2, Bilirubin, and Lactate. The model achieves a mean ROC-AUC of 0.8577 and accuracy of 0.8046, outperforming existing benchmarks.

## ✨ Features
- **Data Preprocessing**: Handles missing data with <35% missingness using linear interpolation.
- **Feature Selection**: Reduces multicollinearity by dropping features with >90% correlation.
- **Model Training**: Implements Random Forest with optimized hyperparameters.
- **Evaluation**: Provides detailed metrics including ROC-AUC, accuracy, precision, recall, and F1-score.

## 📊 Model Evaluation

### Random Forest Model Details
Random Forest (RF) was chosen for its ability to handle high-dimensional datasets with many features. It effectively evaluates feature importance using an ensemble of decision trees to improve predictive performance.

Before training, we analyzed feature correlations and dropped those with >90% correlation to reduce multicollinearity and improve model stability.

### Hyperparameters
| Parameter                          | Value                         |
|------------------------------------|-------------------------------|
| Number of Estimators (`n_estimators`) | 50 trees                     |
| Maximum Depth (`max_depth`)        | 7 to prevent overfitting      |
| Minimum Samples per Split (`min_samples_split`) | 10                |
| Minimum Samples per Leaf (`min_samples_leaf`) | 5                  |
| Maximum Features per Split (`max_features`) | Square root of the total features |
| Bootstrap Sampling                 | Enabled for diversity         |

This configuration ensures the model generalizes well while avoiding overfitting, given the dataset's high dimensionality.

### Cross-Validation Results
- **Mean ROC-AUC Score**: 0.8577  
  The ROC-AUC score measures performance across all classification thresholds. A higher AUC indicates better performance. Our RF model achieved an AUC of 0.8577, showing strong ability to distinguish between septic and non-septic classes.

- **Mean Accuracy**: 0.8046  
  The model correctly predicted outcomes 80.46% of the time, a solid result for this task.

### Classification Report
The classification report provides precision, recall, and F1-score for both septic and non-septic classes:

| Class              | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| Class 0 (Non-Septic) | 0.83      | 0.94   | 0.88     | 1400    |
| Class 1 (Septic)   | 0.79      | 0.54   | 0.64     | 600     |

- **F1-score for Class 0 (Non-Septic)**: 0.88
- **F1-score for Class 1 (Septic)**: 0.64

The F1-score balances precision and recall, showing better performance for non-septic cases due to higher recall.

### Confusion Matrix
The confusion matrix breaks down true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN):

|                    | Predicted: Non-Septic | Predicted: Septic |
|--------------------|-----------------------|-------------------|
| **Actual: Non-Septic** | 1313 (TN)            | 87 (FP)          |
| **Actual: Septic**     | 274 (FN)             | 326 (TP)         |

The model correctly identified 1313 non-septic and 326 septic cases, with 87 false positives and 274 false negatives.

### ROC Curve and AUC Score
The ROC Curve shows the trade-off between True Positive Rate (sensitivity) and False Positive Rate (1-specificity).

- **AUC Score**: 0.8670  
This indicates strong discrimination between septic and non-septic cases.

### Model Comparison
| Metric          | Our Ensemble Model (Optimal Threshold) | Yang et al. (2022) (Ensemble Result) |
|-----------------|----------------------------------------|--------------------------------------|
| Accuracy (ACC)  | 0.81                                   | 0.818                                |
| AUROC           | 0.895                                  | 0.847                                |
| U-Score         | 0.752                                  | 0.425                                |

Our model outperforms Yang et al. (2022) in AUROC and U-Score, showing better discrimination and balanced performance.

## 🔍 Feature Importance
The top features identified by the Random Forest model are:

1. **FiO2 (Fraction of Inspired Oxygen)**: Ranks highest as it assesses oxygen delivery, crucial for detecting respiratory failure like ARDS. The PaO2/FiO2 ratio below 300 mmHg indicates hypoxemia, a key predictor of severe respiratory decline [LITFL](https://litfl.com).
2. **PaCO2 (Partial Pressure of Carbon Dioxide)**: Reflects CO2 removal and respiratory acidosis, common in septic shock, aiding in identifying lung complications [LITFL](https://litfl.com).
3. **Bilirubin_total**: Indicates liver dysfunction, defined as TBIL ≥ 2 mg/dL, linked to poorer outcomes in septic shock, capturing systemic severity [Sun et al. (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7970876/).
4. **Lactate**: Reflects tissue hypoxia, a critical marker of sepsis severity, with higher levels (3.6 mmol/L vs. 2.8 mmol/L, P = 0.031) associated with worse outcomes [Sun et al. (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7970876/).

