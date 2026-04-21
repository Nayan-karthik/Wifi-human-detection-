import joblib
import numpy as np

def predict(signal_value):
    model = joblib.load("models/model.pkl")

    prediction = model.predict([[signal_value]])

    if prediction[0] == 1:
        return "Human Detected"
    else:
        return "No Human"
