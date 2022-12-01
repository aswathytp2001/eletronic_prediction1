"""PRML_Course_Project.ipynb
"""
import sys
import warnings
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

df = pd.read_csv('mytable.csv')
clf = LogisticRegression()

# selected features
selected_features = ['salary', 'Product']

X = df[selected_features]
y = df['purchased']

categorical_cols = [col for col in X.columns if X[col].nunique() < 10 and X[col].dtype == 'object']

numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', clf)
                              ])


my_pipeline.fit(X, y)

pickle.dump(my_pipeline, open('saved.pkl', 'wb'))