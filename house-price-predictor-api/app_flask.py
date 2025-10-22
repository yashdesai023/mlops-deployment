# app_flask.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/house_price_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to House Price Predictor API"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[data["MedInc"], data["HouseAge"], data["AveRooms"], data["AveBedrms"],
                          data["Population"], data["AveOccup"], data["Latitude"], data["Longitude"]]])
    prediction = model.predict(features)
    return jsonify({"predicted_price": round(float(prediction[0]), 2)})

if __name__ == "__main__":
    app.run(debug=True)