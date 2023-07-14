import joblib
import pandas as pd
import numpy as np
import math

modelNumber = 14
damageName = "Chloride Stress Corrosion Cracking"

loaded_model = joblib.load('./models_ml/damages/model'+str(modelNumber)+'.joblib')

def predict(data: dict):
    material = data.material if data.material is not None else 0
    operatingTemperature = data.operatingTemperature if not math.isnan(data.operatingTemperature) else 0
    postWeldHeatTreatment = data.postWeldHeatTreatment if data.postWeldHeatTreatment is not None else 0
    modelFluid = data.modelFluid if data.modelFluid is not None else 0
    waterContains = data.waterContains if not math.isnan(data.waterContains) else 0
    externalEnvironment = data.externalEnvironment if data.externalEnvironment is not None else 0
    pH = data.pH if not math.isnan(data.pH) else 0
    input_data = pd.DataFrame(
                                {
                                    'material': [material], 
                                    'operatingTemperature':[operatingTemperature],
                                    'postWeldHeatTreatment':[postWeldHeatTreatment],
                                    'modelFluid':[modelFluid],
                                    'waterContains':[waterContains],
                                    'externalEnvironment':[externalEnvironment],
                                    'pH':[pH],
                                }
                            )
    
    predicted_damage = loaded_model.predict(input_data)
    predicted_proba = loaded_model.predict_proba(input_data)

    if predicted_damage[0] == modelNumber:
        damage = damageName
    elif predicted_damage[0] == 71:
        damage = "General Corrosion"
    proba_percent = np.max(predicted_proba[0]) * 100
    proba_percent = int(proba_percent)  # Convert to integer
    
    return {"model": modelNumber,"damage": damage, "proba_percent": proba_percent}