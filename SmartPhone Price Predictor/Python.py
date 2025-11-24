
---

# 3) `train_and_save_model.py` (copy this file)

```python
"""
train_and_save_model.py

Train a RandomForest regression pipeline on the Sales.csv dataset and save the trained pipeline to disk.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- CONFIG ----------
DATA_PATH = Path("/mnt/data/Sales.csv")   # <<-- this is your uploaded dataset path
MODEL_OUT = Path("model.pkl")
# ----------------------------

def load_and_clean(path: Path):
    df = pd.read_csv(path)
    # Basic cleaning (drop duplicates)
    df = df.drop_duplicates().reset_index(drop=True)

    # If you used df.dropna earlier outside, you can still impute here:
    # Interpolate numeric rating if present
    if 'Rating' in df.columns:
        df['Rating'] = df['Rating'].interpolate(method='linear')

    return df

def build_pipeline(categorical_features, numerical_features):
    # Transformer for categorical: impute most_frequent -> one-hot
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Numerical imputer (mean)
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipeline, categorical_features),
            ('num', num_pipeline, numerical_features)
        ],
        remainder='drop'  # drop any other columns not specified
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    return model

def main():
    df = load_and_clean(DATA_PATH)

    # Select features and target
    X = df[['Brands', 'Models', 'Memory', 'Camera', 'Storage', 'Discount', 'Original Price']].copy()
    y = df['Selling Price'].copy()

    # Cast numeric columns if necessary
    X['Discount'] = pd.to_numeric(X['Discount'], errors='coerce')
    X['Original Price'] = pd.to_numeric(X['Original Price'], errors='coerce')

    categorical_features = ['Brands', 'Models', 'Memory', 'Camera', 'Storage']
    numerical_features = ['Discount', 'Original Price']

    pipeline = build_pipeline(categorical_features, numerical_features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"R² score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    # Save pipeline
    joblib.dump(pipeline, MODEL_OUT)
    print(f"Saved trained pipeline to {MODEL_OUT.resolve()}")

if __name__ == "__main__":
    main()
