# Pandas is a software library written in Python for data manipulation and analysis.
import pandas as pd
# NumPy is a library which provides support for large, multi-dimensional arrays, matrices and mathematical operations.
import numpy as np
# Matplotlib is a plotting library for python; pyplot offers a MATLAB-style framework for plotting.
import matplotlib.pyplot as plt
# Seaborn is a data visualization library based on matplotlib, allowing users to create complex statistical graphs.
import seaborn as sns

# Preprocessing and transformation
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector as selector

# Provides tools to split data into training and testing sets and perform hyperparameter tuning with cross-validation
from sklearn.model_selection import train_test_split, GridSearchCV

# Shows cross-validated training and test scores for different training set sizes
from sklearn.model_selection import learning_curve

# ConfusionMatrixDisplay is used to plot a confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Pipeline class allows users to chain together multiple preprocessing and modeling steps
from sklearn.pipeline import Pipeline

# SMOTE is a method used to handle class imbalance in datasets by generating samples of minority classes.
from imblearn.over_sampling import SMOTE

class KSIDataLoader:
    def __init__(self, path):
        self.path = path

    def load_and_prepare_data(self):
        # Load the CSV dataset file
        df = pd.read_csv(self.path)

        # Dataset description 
        print("Dataset Dimensions (Rows, Columns):", df.shape)
        print("\nList of Column Names:", df.columns)

        # Display all unique values in a column.
        print("\nUnique values per column:")
        print(df.nunique())

        print("\nData Types & Non-Null (Missing) Value Counts:")
        df.info()

        print("\nSummary Statistics:")
        print(df.describe())

        print("\nExtracting Unique Values from Categorical Columns:")
        for col in df.select_dtypes(include='object'):
            print(f"\n{col}:")
            print(df[col].unique())

        # Identify Missing Values
        print("\n")
        print(df.isnull().sum())

        # Convert DATE to datetime and extract YEAR, MONTH, DAY
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month
        df['WEEKDAY'] = df['DATE'].dt.weekday

        # Replace 'Property Damage Only' with 'Non-Fatal Injury' to create binary classification
        df['ACCLASS'] = df['ACCLASS'].replace('Property Damage O', 'Non-Fatal Injury')

        # Filter only 'Fatal' and 'Non-Fatal Injury'
        df = df[df['ACCLASS'].isin(['Fatal', 'Non-Fatal Injury'])].copy()

        # Fill 'Yes/No' fields with 'No' where there is missing value
        binary_col = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE',
                      'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER',
                      'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL',
                      'DISABILITY']
        df[binary_col] = df[binary_col].fillna('No')

        # Remove irrelevant or unused columns
        remove_col = ['OBJECTID', 'INDEX', 'ACCNUM', 'DATE', 'TIME', 'STREET1',
                      'STREET2', 'OFFSET', 'ACCLOC', 'FATAL_NO', 'PEDTYPE', 
                      'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT', 'CYCCOND', 
                      'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 
                      'x', 'y']

        df = df.drop(columns=[col for col in remove_col if col in df.columns])

        df['ACCLASS'] = df['ACCLASS'].map({'Non-Fatal Injury': 0, 'Fatal': 1})

        print("Fatal vs Non-Fatal Distribution (%):")
        print(df['ACCLASS'].value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

        # Split dataset into x (input features) and y (target collision severity)
        X = df.drop('ACCLASS', axis=1)
        y = df['ACCLASS']

        return df, X, y

    def visualize_distributions(self, df):
        # Visualize key categorical features
        visualize = [
            'YEAR', 'ROAD_CLASS', 'DISTRICT', 'TRAFFCTL', 'VISIBILITY',
            'LIGHT', 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'VEHTYPE',
            'INVTYPE', 'INVAGE', 'DRIVACT', 'DRIVCOND',
        ]
        palette_color = "Set3"

        # Generate count plots for each column
        for col in visualize:
            if col in df.columns:
                n_unique = df[col].nunique()
                plt.figure(figsize = (max(10, min(n_unique, 20)), 5))  # Adjusts width based on # of categories
                sns.countplot(data = df, x = col, hue = col, order = df[col].value_counts().index, palette=palette_color, legend=False)
                plt.title(f'Number of Collisions in Toronto by {col}', fontweight='bold', fontsize = 14)
                plt.xlabel(col, fontweight = 'bold', fontsize = 12)
                plt.ylabel('Count', fontweight = 'bold', fontsize = 12)
                plt.xticks(rotation=45, ha = 'right', fontsize=10)
                plt.tight_layout()
                plt.show()
