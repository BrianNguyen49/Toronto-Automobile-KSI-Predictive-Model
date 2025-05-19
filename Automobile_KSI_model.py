from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

from utils import evaluate_model, plot_learning_curve, plot_all_feature_importances, plot_feature_importances

# Logistic Regression Model

def logistic_regression(X, y):
    # Define Preprocessing pipelines
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, selector(dtype_include='object')),
        ('num', num_pipeline, selector(dtype_exclude='object'))
    ])

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Split preprocessed data into stratified training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, stratify=y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance classes
    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Logistic Regression setup
    logreg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=500, solver='lbfgs', penalty='l2'))
    ])

    param_grid = {
        'logreg__C': [0.01, 0.1, 1, 10],
        'logreg__solver': ['lbfgs', 'liblinear'],
        'logreg__penalty': ['l2']
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=logreg_pipeline,
        param_grid=param_grid,
        scoring='recall',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_bal, y_train_bal)

    # Evaluate model
    model = grid_search.best_estimator_
    
    evaluate_model(model, X_test, y_test, X_full=X_processed, y_full=y, title="Logistic Regression")

    plot_learning_curve(model, X_processed, y, title="Learning Curve for Logistic Regression")

# Random Forest Classifier

def random_forest(X, y):
    # Define Preprocessing pipelines
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, selector(dtype_include='object')),
        ('num', num_pipeline, selector(dtype_exclude='object'))
    ])

    # Apply preprocessing
    X_rf_processed = preprocessor.fit_transform(X)

    rf_temp = RandomForestClassifier(random_state=13, n_jobs=-1, class_weight='balanced')
    rf_temp.fit(X_rf_processed, y)

    importances = rf_temp.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_temp.estimators_], axis=0)
    feature_names = X.columns
    top_n = 12
    important_indices = np.argsort(importances)[-top_n:]
    X_selected = X_rf_processed[:, important_indices]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=13)

    X_train_bal, y_train_bal = SMOTE(random_state=13).fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10],
        'min_samples_split': [2],
        'max_features': ['sqrt']
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=13, class_weight='balanced'),
        param_grid=param_grid,
        scoring='recall',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_bal, y_train_bal)

    model = grid.best_estimator_

    plot_all_feature_importances(importances, std, feature_names=X.columns)

    plot_feature_importances(importances, std, feature_names=X.columns, top_n=12)

    evaluate_model(model, X_test, y_test, X_full=X_selected, y_full=y, title="Random Forest", threshold=0.43)

    plot_learning_curve(model, X_selected, y, title="Learning Curve for Random Forest")


# XGBoost Model

def xgboost(X, y):
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, selector(dtype_include='object')),
        ('num', num_pipeline, selector(dtype_exclude='object'))
    ])

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=13)

    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [6, 10],
        'learning_rate': [0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [1.0]
    }

    grid_search = GridSearchCV(
        estimator=XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42),
        param_grid=param_grid,
        scoring='recall',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_bal, y_train_bal)

    model = grid_search.best_estimator_

    evaluate_model(model, X_test, y_test, X_full=X_processed, y_full=y, title="XGBoost", threshold=0.2)

    plot_learning_curve(model, X_processed, y, title="Learning Curve for XGBoost")


# LightGBM Model

def lightgbm(X, y):
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, selector(dtype_include='object')),
        ('num', num_pipeline, selector(dtype_exclude='object'))
    ])

    X_processed = pd.DataFrame(
        preprocessor.fit_transform(X),
        columns=preprocessor.get_feature_names_out()
    )

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=13)

    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [6, 10, -1],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=LGBMClassifier(objective='binary', class_weight='balanced', random_state=42, verbose=-1),
        param_grid=param_grid,
        scoring='recall',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_bal, y_train_bal)

    model = grid_search.best_estimator_

    evaluate_model(model, X_test, y_test, X_full=X_processed, y_full=y, title="LightGBM", threshold=0.317)

    plot_learning_curve(model, X_processed, y, title="Learning Curve for LightGBM")
