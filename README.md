# Predicting Automobile-related Killed and Seriously Injured (KSI) Incidents in Toronto

The aim of this project is to develop a machine learning model to analyze and identify trends in automobile-related KSI incidents within Toronto. It utilizes the Automobile-related KSI Collisions dataset (2006-2024) provided by Toronto Police Service through their [Public Safety Data Portal](https://data.torontopolice.on.ca/datasets/TorontoPS::automobile-ksi/about) to predict the likelihood of fatal or non-fatal Automobile-related KSI occurrences in Toronto.

## Setup and Requirements

- **Python Version**: 3.13  
- **Environment**: Jupyter Notebook  
- **Packages**:
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib
  - seaborn
  - imbalanced-learn (SMOTE)
  - XGBoost

To reproduce the project, pelase install the following dependencies: pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

## Findings

- Top predictive features for the Random Forest and model included `INJURY`, `LATITUDE`, `LONGITUDE`, `YEAR`, `MONTH`, `DIVISION`, `IMPACTYPE`, and `INVTYPE`
- After reclassifying 'Property Damage Only' as 'Non-Fatal Injury', approximately 13.27% of collisions are fatal and 86.73% are non-fatal.
- The ACCLASS column was used as the target variable, with 'Fatal' encoded as 0 and 'Non-Fatal Injury' as 1.
- Several sparse (data with many gaps) columns (e.g., SPEEDING, DISABILITY, ALCOHOL) had mostly missing values that were filled with 'No' to maintain consistency.
- Some columns with highly specific location information or too many unique values (e.g., STREET1, LATITUDE, HOOD_140) were removed to reduce noise and dimensionality in the model.

## Model Evaluation Results

| Model              | Accuracy  | Precision | F1 Score | Recall   | ROC AUC |
|--------------------|-----------|-----------|----------|----------|---------|
| **Random Forest**  | 93.63%    | 94.07%    | 96.42%   | 98.89%   | 0.926   |
| **Logistic Reg.**  | 80.04%    | 94.67%    | 87.64%   | 81.58%   | 0.853   |
| **XGBoost**        | 94.39%    | 94.43%    | 96.85%   | 99.39%   | 0.946   |

### Key Insights

Random Forest and XGBoost both delivered strong performance, especially in identifying fatal collisions, with high accuracy, recall and F1 scores Logistic Regression performed well in terms of precision but lagged behind in terms of recall and overall accuracy, making it less reliable for detecting fatal outcomes. XGBoost outperformed the other models across all evaluation metrics, slightly outperforming Random Forest in all categories. Therefore we can conclude that XGBoost is the most effective model for distinguishing between fatal and non-fatal cases.
