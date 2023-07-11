import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.predict import DamagePredictionRequest, DamagePredictionResponse
from fastapi import APIRouter

loaded_model = joblib.load('./models_ml/damages/model1.joblib')

router = APIRouter()

@router.post("/")
def predict_damage(request: DamagePredictionRequest):
    material = request.material
    operating_temperature = request.operating_temperature
    amine_contains = request.amine_contains

    # Perform preprocessing on the input values
    input_data = pd.DataFrame({'Material': [material], 'Operating Temperature': [operating_temperature], 'Amine Contains': [amine_contains]})

    # Make prediction using the loaded model
    predicted_damage = loaded_model.predict(input_data)
    predicted_proba = loaded_model.predict_proba(input_data)
    if predicted_damage[0] == 2:
        damage = "Amine Corrosion"
    elif predicted_damage[0] == 71:
        damage = "General Corrosion"
    proba_percent = np.max(predicted_proba[0]) * 100
    proba_percent = int(proba_percent)  # Convert to integer

    return DamagePredictionResponse(damage=damage, probability=proba_percent)