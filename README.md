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

## Findings

Based on the outcome of the evaluation of the three models, approximately 90% of the collisions in the dataset are non-fatal, with the remaining 10% being fatal.
Many binary fields (e.g., SPEEDING, DISABILITY) provided sparse but useful signals for classification. Categorical variables like LIGHT, VISIBILITY, ROAD_CLASS showed significant variation across fatal vs non-fatal cases

2. Logistic Regression

Preprocessing: One-hot encoding for categoricals, MinMax scaling

SMOTE applied to training data

GridSearchCV parameters:

C: [0.1, 1, 10]

Evaluation Metrics:

Accuracy: 80.19%
Precision: 94.65%
Recall: 81.78%
F1 Score: 87.75%
ROC AUC: 0.855

3. XGBoost Classifier

Preprocessing: Ordinal encoding for categoricals, MinMax scaling

SMOTE applied to training data

GridSearchCV parameters:

n_estimators: [100, 200]
max_depth: [6, 8, 10]
learning_rate: [0.05, 0.1]
subsample: [0.8, 1.0]
colsample_bytree: [0.8, 1.0]

Evaluation Metrics:
Accuracy: ~95.81%
Precision: ~95.57%
Recall: ~99.79%
F1 Score: ~97.64%
ROC AUC: ~0.965
