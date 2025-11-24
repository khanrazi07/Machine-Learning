# Smartphone Price Prediction Model

A machine learning project that predicts smartphone selling price from product specifications (brand, model, memory, storage, rating, discount, original price). Includes data preprocessing, encoding of categorical features, a Random Forest regression pipeline, model saving, and prediction examples.

**Dataset:** Provided as `Sales.csv`. (In this repo demo the data path used during development: `/mnt/data/Sales.csv`.)

---

## Features used
- Brands (categorical)
- Models (categorical)
- Memory (categorical)
- Camera (categorical/text)
- Storage (categorical)
- Discount (numeric)
- Original Price (numeric)

## What you get
- `train_and_save_model.py` — full training script, evaluation, and saves `model.pkl`
- `predict_example.py` — example how to load the model and predict
- `requirements.txt` — required Python packages
- `model.pkl` — (create by running the training script)
- Instructions to deploy as an API or Streamlit app

---

## Quick start

1. Clone the repo:
```bash
git clone <your-repo-url>
cd smartphone-price-prediction
