# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

# Load trained model from MLflow
model = mlflow.sklearn.load_model("runs:/6202ddbc4a2f4cb0b7a3438a9faec91d/linear_regression_model")  

# Create FastAPI app
app = FastAPI()

# Define input schema
class InputData(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"prediction": prediction[0]}
