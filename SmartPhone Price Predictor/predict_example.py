"""
predict_example.py

Load saved pipeline 'model.pkl' and run a sample prediction.
"""

import pandas as pd
import joblib
from pathlib import Path

MODEL_IN = Path("model.pkl")

def predict_price_example():
    # Load pipeline
    pipeline = joblib.load(MODEL_IN)

    # Example input - change values to test
    input_df = pd.DataFrame([{
        'Brands': 'Samsung',
        'Models': 'Galaxy M31S',
        'Memory': '8 GB',
        'Camera': 'Yes',     # or '48MP' depending on your dataset encoding
        'Storage': '128 GB',
        'Discount': 1669,
        'Original Price': 20999
    }])

    pred = pipeline.predict(input_df)
    print("Predicted selling price:", round(float(pred[0]), 2))

if __name__ == "__main__":
    predict_price_example()
