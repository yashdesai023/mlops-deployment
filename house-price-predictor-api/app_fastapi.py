# app_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title=" House Price Predictor API")

# Load model
model = joblib.load("model/house_price_model.pkl")

# Define input schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def root():
    return {"message": "Welcome to House Price Predictor API"}

@app.post("/predict")
def predict_price(data: HouseFeatures):
    features = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                          data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    prediction = model.predict(features)
    return {"predicted_price": round(float(prediction[0]), 2)}