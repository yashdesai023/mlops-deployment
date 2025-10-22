# app_fastapi.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="House Price Predictor API")

# Mount the 'static' directory to serve CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model = joblib.load("model/house_price_model.pkl")

# Define input schema (no changes here)
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
    """Serves the frontend HTML page."""
    return FileResponse("templates/index.html")

@app.post("/predict")
def predict_price(data: HouseFeatures):
    """Receives data from the form and returns a prediction."""
    features = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                          data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    
    prediction = model.predict(features)
    
    return {"predicted_price": round(float(prediction[0]), 2)}

# Note: To run this file, you would use:
# uvicorn app_fastapi:app --reload