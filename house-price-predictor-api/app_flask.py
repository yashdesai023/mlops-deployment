# app_flask.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model/house_price_model.pkl")

@app.route("/", methods=["GET"])
def home():
    """Serves the frontend HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Receives data from the form and returns a prediction."""
    data = request.get_json()
    features = np.array([[data["MedInc"], data["HouseAge"], data["AveRooms"], data["AveBedrms"],
                          data["Population"], data["AveOccup"], data["Latitude"], data["Longitude"]]])
    
    prediction = model.predict(features)
    
    return jsonify({"predicted_price": round(float(prediction[0]), 2)})

if __name__ == "__main__":
    app.run(debug=True)