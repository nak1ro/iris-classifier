# app/predict.py
import joblib
import numpy as np

# Load the saved model
model_data = joblib.load("model.joblib")

model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]
target_names = model_data["target_names"]


def predict(features):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    species = target_names[prediction]

    return species
