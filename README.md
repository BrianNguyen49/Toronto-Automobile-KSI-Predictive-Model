# Predicting Automobile-related Killed and Seriously Injured (KSI) Incidents in Toronto

The aim of this project is to develop a machine learning model to analyze and identify trends in automobile-related KSI incidents within Toronto. It utilizes the Automobile-related KSI Collisions dataset (2006-2024) provided by Toronto Police Service through their [Public Safety Data Portal](https://data.torontopolice.on.ca/datasets/TorontoPS::automobile-ksi/about) to predict the likelihood of fatal or non-fatal Automobile-related KSI occurrences in Toronto.

## Exploratory Data Analysis (EDA)

The goal of the EDA phase is to gain insight into the dataset and identify its structure, variable relationships, feature distributions, and potential outliers. This step ensures the data is suitable for modeling and helps highlight the most informative variables that may influence the outcome of automobile-related KSI  collisions in Toronto.

EDA actions performed on the dataset include:

- Checking dataset dimensions and column names
- Reviewing data types, missing values, and summary statistics
- Converting the `DATE` column to datetime and extracting `YEAR`, `MONTH`, and `WEEKDAY` 
- Inspecting unique values in categorical columns
- Visualizing distributions of key categorical features

## Data Cleaning

The goal of the data cleaning phase is to improve data quality by identifying, correcting and transforming inconsistencies in the dataset. This step ensures that the data is accurate and consistent, helping to reduce bias and uncertainty when training supervised learning models.

Data cleaning techniques applied:

- Reclassifying "Property Damage Only" entries as "Non-Fatal Injury" to support binary classification
- Filtering the dataset to include only "Fatal" and "Non-Fatal Injury" cases
- Filling missing values in binary categorical columns (e.g., PEDESTRIAN, ALCOHOL) with "No"
- Dropping irrelevant or redundant columns
- Applying label encoding to convert the target column (ACCLASS) into a binary numeric format

## Data Preprocessing

The goal of the data preprocessing phase is to prepare and transform the raw data into a format suitable for machine learning models. This phase includes feature scaling, feature selection, encoding categorical variables, and balancing the classes using SMOTE.

Preprocessing steps performed:

- Splitting the dataset into features `X` and target variable `y`
- Encoding categorical variables using OneHotEncoding or OrdinalEncoding depending on the model used
- Applied data imputation to fill missing values in categorical and numerical features
- Scaled numerical features using MinMaxScaler as part of a numerical pipeline within a ColumnTransformer to proportionally scale values to a consistent range.
- Balancing the dataset using SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance prior to model training

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

## Installation

To set up your environment for running the notebook, make sure Python is installed on your system along with the required libraries. You can install all dependencies using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost
```

## Key Insights

- Top predictive features for the Random Forest and model included `INJURY`, `LATITUDE`, `LONGITUDE`, `YEAR`, `MONTH`, `DIVISION`, `IMPACTYPE`, and `INVTYPE`
- After reclassifying 'Property Damage Only' as 'Non-Fatal Injury', approximately 13.27% of collisions are fatal and 86.73% are non-fatal.
- The ACCLASS column was used as the target variable, with 'Fatal' encoded as 0 and 'Non-Fatal Injury' as 1.
- Several sparse (data with many gaps) columns (e.g., SPEEDING, DISABILITY, ALCOHOL) had mostly missing values that were filled with 'No' to maintain consistency.
- Some columns with highly specific location information or too many unique values (e.g., STREET1, LATITUDE, HOOD_140) were removed to reduce noise and dimensionality in the model.

## Confusion Matrix and Metric Calculations

<div align="center">
<img src="https://github.com/user-attachments/assets/fb3c0994-9e99-45d7-9ee2-50c5dbc756d1" width="30%" alt="Logistic Regression Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/b02675b9-1893-4242-93c9-fc9ab5c2d8f6" width="30%" alt="Random Forest Confusion Matrix"/>
<img src="https://github.com/user-attachments/assets/58255c3e-4ab1-4b21-9449-4502ff2cd889" width="30%" alt="XGBoost Confusion Matrix"/>
</div>


To better understand model performance, we used the confusion matrix to manually calculate model performance metrics including accuracy, precision, F1, and recall score. In order to determine how well a classification model is performing, we plot a confusion matrix which summarizes prediction results by comparing the actual labels with the predicted labels. The confusion matrix breaks down the predictions into four categories: true positive, true negative, false positive, false positive. 

In this project, 'Fatal' collisions are encoded as 0 and 'Non-Fatal' as 1. Therefore:

- True Positive (TP): the model correctly predicted positive cases (e.g., predicted outcome was non-fatal)
- True Negative (TN): the model correctly predicted negative cases (e.g., predicted outcome was fatal)
- False Positive (FP): the model incorrectly predicted positive cases (e.g., non-fatal incorrectly predicted as fatal cases)
- False Negative (FN): the model incorrectly predicted negative cases (e.g., missed fatal cases)

### Confusion Matrix Summary

| Model                  | True Negatives (TN) | False Positives (FP)  | False Negatives (FN)  | True Positives (TP) |
|------------------------|---------------------|-----------------------|-----------------------|---------------------|
| **Logistic Regression**| 320                 | 137                   | 550                   | 2436                |
| **Random Forest**      | 271                 | 186                   | 33                    | 2953                |
| **XGBoost**            | 282                 | 175                   | 18                    | 2968                |

Using these counts, we calculate the model performance metrics using the following formulas:

- **Accuracy score**  = (TP + TN) / (TP + FN + TN + FP)
- **Precision score** = TP / (TP + FP)
- **F1 Score score**  = 2 * (Precision * Recall) / (Precision + Recall)
- **Recall score**    = TP / (FN + TP)

### Logistic Regression  
- TP = 2436, TN = 320, FP = 137, FN = 550  
- Accuracy  = (2436 + 320) / 3443 = **0.800**  
- Precision = 2436 / (2436 + 137) = **0.946**
- F1 Score  = 2 * (0.947 * 0.816) / (0.947 + 0.816) = **0.876**
- Recall    = 2436 / (2436 + 550) = **0.815**  

### Random Forest  
- TP = 2953, TN = 271, FP = 186, FN = 33  
- Accuracy  = (2953 + 271) / 3443 = **0.936**  
- Precision = 2953 / (2953 + 186) = **0.940**
- F1 Score  = 2 * (0.941 * 0.989) / (0.941 + 0.989) = **0.964**
- Recall    = 2953 / (2953 + 33) = **0.989**  

### XGBoost  
- TP = 2968, TN = 282, FP = 175, FN = 18  
- Accuracy  = (2968 + 282) / 3443 = **0.943**  
- Precision = 2968 / (2968 + 175) = **0.944**
- F1 Score  = 2 * (0.944 * 0.993) / (0.944 + 0.993) = **0.968**
- Recall    = 2968 / (2968 + 18) = **0.993**  

## Model Evaluation Results

| Model                   | Accuracy  | Precision | F1 Score | Recall   | ROC AUC |
|-------------------------|-----------|-----------|----------|----------|---------|
| **Logistic Regresion**  | 80.04%    | 94.67%    | 87.64%   | 81.58%   | 0.853   |
| **Random Forest**       | 93.63%    | 94.07%    | 96.42%   | 98.89%   | 0.926   |
| **XGBoost**             | 94.39%    | 94.43%    | 96.85%   | 99.39%   | 0.946   |

### Results Summary

Random Forest and XGBoost both delivered strong performance, especially in identifying fatal collisions, with high accuracy, recall and F1 scores Logistic Regression performed well in terms of precision but lagged behind in terms of recall and overall accuracy, making it less reliable for detecting fatal outcomes. XGBoost outperformed the other models across all evaluation metrics, slightly outperforming Random Forest in all categories. Therefore we can conclude that XGBoost is the most effective model for distinguishing between fatal and non-fatal cases.
