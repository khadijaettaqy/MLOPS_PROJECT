"""
inference.py

Inference module for heart disease prediction.
Loads the trained Pipeline (scaler + classifier) and makes predictions.
"""

import joblib
import pandas as pd
from pathlib import Path

_DEFAULT_MODEL = Path(__file__).resolve().parent / "heart_model.pkl"


class HeartDiseasePredictor:
    """
    Prediction pipeline for heart disease classification.
    Loads a single sklearn Pipeline that includes preprocessing and model.
    """

    FEATURE_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    def __init__(self, model_path=None):
        """Load the trained pipeline."""
        if model_path is None:
            model_path = _DEFAULT_MODEL
        self.pipeline = joblib.load(model_path)

    def predict(self, input_data: dict) -> dict:
        """
        Make a prediction from a dictionary of features.

        Args:
            input_data: dict with feature names as keys

        Returns:
            dict with prediction and confidence
        """
        df = pd.DataFrame([input_data])[self.FEATURE_NAMES]

        prediction = int(self.pipeline.predict(df)[0])
        probability = float(self.pipeline.predict_proba(df)[0][1])

        return {
            'prediction': prediction,
            'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'confidence': round(probability, 4),
            'risk_level': self._get_risk_level(probability)
        }

    def _get_risk_level(self, probability: float) -> str:
        """Categorize risk level based on probability."""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"

    def predict_batch(self, input_list: list) -> list:
        """Make predictions for multiple inputs."""
        return [self.predict(item) for item in input_list]


def get_sample_input():
    """Return a sample input for testing."""
    return {
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 130,
        'chol': 250,
        'fbs': 0,
        'restecg': 1,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 1.5,
        'slope': 1,
        'ca': 0,
        'thal': 3
    }


if __name__ == "__main__":
    print("=" * 50)
    print("HEART DISEASE PREDICTION - INFERENCE TEST")
    print("=" * 50)

    predictor = HeartDiseasePredictor()
    sample = get_sample_input()

    print(f"\nInput: {sample}")
    result = predictor.predict(sample)
    print(f"\nPrediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk Level: {result['risk_level']}")
