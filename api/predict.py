"""
Vercel Serverless Function — Injury Prediction API
"""
from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'injury_model.pkl')
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    "Age", "BMI", "Total Distance", "Sprint Count", "Acceleration Load",
    "ACWR", "Yo-Yo Score", "Jump Height", "Previous Injuries", "Minutes Played",
]


@app.route('/api/predict', methods=['POST'])
def predict():
    """Accept 10 feature values and return injury prediction."""
    try:
        data = request.get_json(force=True)
        inputs = data.get('inputs', [])
        if len(inputs) != 10:
            return jsonify({'error': 'Exactly 10 input features required'}), 400

        X = np.array(inputs, dtype=float).reshape(1, -1)
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0].tolist()
        importances = model.feature_importances_.tolist()

        return jsonify({
            'prediction': prediction,
            'label': 'HIGH RISK' if prediction == 1 else 'LOW RISK',
            'risk_probability': round(probabilities[1] * 100, 1),
            'safe_probability': round(probabilities[0] * 100, 1),
            'feature_importances': dict(zip(FEATURE_NAMES, importances)),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
