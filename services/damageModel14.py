import joblib
import pandas as pd
import numpy as np

modelNumber = 14
damageName = "Chloride Stress Corrosion Cracking"

loaded_model = joblib.load('./models_ml/damages/model'+str(modelNumber)+'.joblib')

def predict(data: dict):
    material = data.material
    operatingTemperature = data.operatingTemperature
    postWeldHeatTreatment = data.postWeldHeatTreatment
    modelFluid = data.modelFluid
    waterContains = data.waterContains
    externalEnvironment = data.externalEnvironment if data.externalEnvironment is not None else 0
    pH = data.pH

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