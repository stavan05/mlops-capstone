from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Boston Housing Price Predictor")

MODEL_PATH = "models/linear_regression.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")
model = joblib.load(MODEL_PATH)

class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.get("/")          # 👈 Add this
def root():
    return {"message": "Boston Housing Price Predictor API is running 🚀"}

@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_price": prediction}
