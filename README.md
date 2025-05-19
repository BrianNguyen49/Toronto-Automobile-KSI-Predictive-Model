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
  - LightGBM

## Installation

To set up your environment for running the notebook, make sure Python is installed on your system along with the required libraries. You can install all dependencies using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost lightgbm
```

## Key Insights

- Top predictive features for the Random Forest and model included `INJURY`, `LATITUDE`, `LONGITUDE`, `YEAR`, `MONTH`, `DIVISION`, `IMPACTYPE`, and `INVTYPE`
- After reclassifying 'Property Damage Only' as 'Non-Fatal Injury', approximately 13.27% of collisions are fatal and 86.73% are non-fatal.
- The ACCLASS column was used as the target variable, with 'Fatal' encoded as 0 and 'Non-Fatal Injury' as 1.
- Several sparse (data with many gaps) columns (e.g., SPEEDING, DISABILITY, ALCOHOL) had mostly missing values that were filled with 'No' to maintain consistency.
- Some columns with highly specific location information or too many unique values (e.g., STREET1, LATITUDE, HOOD_140) were removed to reduce noise and dimensionality in the model.

## Confusion Matrix and Metric Calculations

<table align="center">
  <tr>
    <td align="center"><strong>Logistic Regression</strong><br>
      <img src="https://github.com/user-attachments/assets/29059ff0-b173-4256-abff-d099b464dce1" width="360"/>
    </td>
    <td align="center"><strong>Random Forest</strong><br>
      <img src="https://github.com/user-attachments/assets/192a68d5-ec35-471a-8dd1-65094d48227a" width="360"/>
    </td>
  </tr>
  <tr>
    <td align="center"><strong>XGBoost</strong><br>
      <img src="https://github.com/user-attachments/assets/d6f49bd5-c816-4877-bcbc-5100d1e2d191" width="360"/>
    </td>
    <td align="center"><strong>LightGBM</strong><br>
      <img src="https://github.com/user-attachments/assets/7182d4e9-7c9c-4e9a-9fff-a9e572f9d032" width="360"/>
    </td>
  </tr>
</table>

To better understand model performance, we used the confusion matrix to manually calculate model performance metrics including accuracy, precision, F1, and recall score. In order to determine how well a classification model is performing, we plot a confusion matrix which summarizes prediction results by comparing the actual labels with the predicted labels. The confusion matrix breaks down the predictions into four categories: true positive, true negative, false positive, false positive. 

In this project, 'Fatal' collisions are encoded as 1 and 'Non-Fatal' as 0. Therefore:

- True Positive (TP): the model correctly predicted positive cases (e.g., predicted outcome was non-fatal)
- True Negative (TN): the model correctly predicted negative cases (e.g., predicted outcome was fatal)
- False Positive (FP): the model incorrectly predicted positive cases (e.g., non-fatal incorrectly predicted as fatal cases)
- False Negative (FN): the model incorrectly predicted negative cases (e.g., missed fatal cases)

### Confusion Matrix Summary

| Model                  | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|------------------------|---------------------|-----------------------|-----------------------|---------------------|
| **Logistic Regression**| 2428                | 561                   | 137                   | 320                 |
| **Random Forest**      | 2951                | 38                    | 150                   | 307                 |
| **XGBoost**            | 2929                | 60                    | 88                    | 369                 |
| **LightGBM**           | 2949                | 40                    | 103                   | 354                 |


Using these counts, we calculate the model performance metrics using the following formulas:

- **Accuracy score**  = (TP + TN) / (TP + FN + TN + FP)
- **Precision score** = TP / (TP + FP)
- **F1 Score score**  = 2 * (Precision * Recall) / (Precision + Recall)
- **Recall score**    = TP / (FN + TP)

### Logistic Regression  
- TP = 320, TN = 2428, FP = 561, FN = 137  
- Accuracy  = (320 + 2428) / 3446 = **0.797**  
- Precision = 320 / (320 + 561) = **0.363**  
- F1 Score  = 2 * (0.363 * 0.700) / (0.363 + 0.700) = **0.478**  
- Recall    = 320 / (320 + 137) = **0.700**  

### Random Forest  
- TP = 307, TN = 2951, FP = 38, FN = 150  
- Accuracy  = (307 + 2951) / 3446 = **0.945**  
- Precision = 307 / (307 + 38) = **0.889**  
- F1 Score  = 2 * (0.890 * 0.672) / (0.890 + 0.672) = **0.765**  
- Recall    = 307 / (307 + 150) = **0.671**  

### XGBoost  
- TP = 369, TN = 2929, FP = 60, FN = 88  
- Accuracy  = (369 + 2929) / 3446 = **0.957**  
- Precision = 369 / (369 + 60) = **0.860**  
- F1 Score  = 2 * (0.860 * 0.807) / (0.860 + 0.807) = **0.832**  
- Recall    = 369 / (369 + 88) = **0.807**  

### LightGBM  
- TP = 354, TN = 2949, FP = 40, FN = 103  
- Accuracy  = (354 + 2949) / 3446 = **0.958**  
- Precision = 354 / (354 + 40) = **0.898**  
- F1 Score  = 2 * (0.899 * 0.775) / (0.899 + 0.775) = **0.832**  
- Recall    = 354 / (354 + 103) = **0.774**  

## Model Evaluation Results

| Model                   | Accuracy | Precision | F1 Score | Recall  | ROC AUC |
|-------------------------|----------|-----------|----------|---------|---------|
| **Logistic Regression** | 79.74%   | 36.32%    | 47.83%   | 70.02%  | 0.846   |
| **Random Forest**       | 94.54%   | 88.98%    | 76.55%   | 67.17%  | 0.943   |
| **XGBoost**             | 95.70%   | 86.01%    | 83.29%   | 80.74%  | 0.955   |
| **LightGBM**            | 95.85%   | 89.84%    | 83.19%   | 77.46%  | 0.952   |

**Note:** The evaluation metrics in this table were computed using model-specific probability thresholds

## ROC Curve & AUC Graph

<p align="center">
  <img src="https://github.com/user-attachments/assets/1f894424-4269-465d-9121-08220dbd3324" width="360" />
  <img src="https://github.com/user-attachments/assets/9c28fed2-fda3-46af-a8e6-e41ba817fdf7" width="360" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/0fcfdc13-6ce5-4266-9017-5866b85cdc82" width="360" />
  <img src="https://github.com/user-attachments/assets/4ae7d8f5-4e35-4298-ae59-a52f2e6da52b" width="360" />
</p>

The ROC curve is a visual representation which illustrates each model's performance to distinguish between fatal and non-fatal automobile-related KSI collisions across all thresholds. The ROC curve is drawn by calculating the true positive rate (TPR) and false positive rate (FPR) at every possible threshold, then graphing TPR over FPR. On the other hand, the AUC (Area Under the Curve) measures how well a model ranks predictions, specifically its ability to assign higher probabilities to positive cases than to negative ones. As obersved by the ROC curve plot graphs for each supervised learning model, XGBoost achieved the highest AUC of 0.96, followed by LightGBM which attained an AUC of 0.95. The XGBoost ROC curve indicates higher sensitivity (TPR) among other models while also minimizing false positives. This concludes that XGBoost is the best performing model at correctly classifying observations into categories (distinguishing between fatal and non-fatal collisions).

### Learning Curve Plot Graph

<table align="center">
  <tr>
    <td align="center"><strong>Logistic Regression</strong><br>
      <img src="https://github.com/user-attachments/assets/42b06f91-e855-4072-ba6c-e710c42c4914" width="360"/>
    </td>
    <td align="center"><strong>Random Forest</strong><br>
      <img src="https://github.com/user-attachments/assets/c0d7c30c-85cb-4549-abca-48942aff464d" width="360"/>
    </td>
  </tr>
  <tr>
    <td align="center"><strong>XGBoost</strong><br>
      <img src="https://github.com/user-attachments/assets/be381bbd-0d4c-4db3-9645-f8e325e32015" width="360"/>
    </td>
    <td align="center"><strong>LightGBM</strong><br>
      <img src="https://github.com/user-attachments/assets/de4593fe-63df-4d06-af8b-5a512fa19200" width="360"/>
    </td>
  </tr>
</table>

As you can see from the plot graphs, the Logistic Regression learning curve reflects moderate bias but generalizes well when dealing with new instances. The graph shows stable test accuracy around 0.89 and a training score of approximately 0.92, indicating low variance. In contrast, Random Forest, XGBoost, and LightGBM all exhibit low bias but high variance and overfitting, with perfect training scores (1.00) but declining test accuracies of ~0.71, ~0.70, and ~0.62 respectively,  which tells us that these models has poor time generalizing data not part of the initial training data.

### Results Summary

Random Forest and XGBoost both delivered strong performance, especially in identifying fatal collisions, with high accuracy, recall and F1 scores Logistic Regression performed well in terms of precision but lagged behind in terms of recall and overall accuracy, making it less reliable for detecting fatal outcomes. XGBoost outperformed the other models across all evaluation metrics, slightly outperforming Random Forest in all categories. Therefore we can conclude that XGBoost is the most effective model for distinguishing between fatal and non-fatal cases.

LightGBM and XGBoost both delivered strong performance, especially in accurately identifying fatal collisions, with high scores across accuracy, precision, recall, and F1 metrics. Random Forest also performed well in most areas, however slightly underperformed in recall compared to all other models. Logistic Regression performed most poorly across almost every metric, making it less suitable for detecting fatal outcomes.Among all models, XGBoost consistently outperformed the others across most evaluation metrics, with recall being the most impactful metric. The model successfully identified over 80% of actual fatal collisions, making XGBoost the most effective choice for distinguishing between fatal and non-fatal cases in the context of automobile-related KSI collisions.
