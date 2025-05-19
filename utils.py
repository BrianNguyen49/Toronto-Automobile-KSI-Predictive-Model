# Pandas is a software library written in Python for data manipulation and analysis.
import pandas as pd
# NumPy is a library which provides support for large, multi-dimensional arrays, matrices and mathematical operations.
import numpy as np
# Matplotlib is a plotting library for python; pyplot offers a MATLAB-style framework for plotting.
import matplotlib.pyplot as plt
# ConfusionMatrixDisplay is used to plot a confusion matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
# Provides tools to split data into training and testing sets and perform hyperparameter tuning with cross-validation
from sklearn.model_selection import cross_val_score, learning_curve

# Function to plot all feature importances using impurity-based importance from tree-based models
def plot_all_feature_importances(importances, std, feature_names, title="Feature Importances"):
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Decrease in Impurity", fontsize=12, fontweight='bold')
    ax.set_xlabel("Features", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Function to plot top N important features from a model based on impurity importance
def plot_feature_importances(importances, std, feature_names, top_n=12, title="Top 12 Important Features"):
    forest_importances = pd.Series(importances, index=feature_names)
    top_features = forest_importances.nlargest(top_n)
    top_std = std[np.argsort(importances)[-top_n:]]

    fig, ax = plt.subplots(figsize=(8, 6))
    top_features.plot.bar(yerr=top_std, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Decrease in Impurity", fontsize=12, fontweight='bold')
    ax.set_xlabel("Features", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Function to evaluate a trained classification model using key metrics and visualize its ROC curve
def evaluate_model(model, X_test, y_test, X_full=None, y_full=None, title="Model", threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {title} Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    # Compute a confusion matrix using sckit-learn to evaluate the accuracy of a classification
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='plasma', colorbar=True, values_format='.0f')
    plt.title(f"Confusion Matrix for {title}", fontsize=14, fontweight='bold')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

    if X_full is not None and y_full is not None:
        cv_scores = cross_val_score(model, X_full, y_full, cv=3, scoring='accuracy')
        print("\nCross-Validation Scores:", cv_scores)
        print("Mean CV Accuracy:", np.mean(cv_scores))
        print("Standard Deviation:", np.std(cv_scores))

    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC Score:", roc_auc)

    # Generate the ROC curve to evaluate the model's performance and ability to distinguish between classes at various decision thresholds
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#4c72b0", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    plt.title(f"ROC Curve for {title} Classifier", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"roc_curve_{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

# Function to plot the learning curve of a trained model which shows training and cross-validation accuracy across varying training set sizes
def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=3, scoring="accuracy"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Number of samples in the training set", fontsize=12, fontweight='bold')
    plt.ylabel(scoring.capitalize(), fontsize=12, fontweight='bold')
    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Test Score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"learning_curve_{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()
