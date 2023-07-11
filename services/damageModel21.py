import joblib
import pandas as pd
import numpy as np

modelNumber = 21
damageName = "Hydrogen Embrittlement"

loaded_model = joblib.load('./models_ml/damages/model'+str(modelNumber)+'.joblib')

def predict(data: dict):
    operatingTemperature = data.operatingTemperature
    postWeldHeatTreatment = data.postWeldHeatTreatment
    hydrogenContains = data.hydrogenContains
    MBHW = data.MBHW

    input_data = pd.DataFrame(
                                {
                                    'operatingTemperature':[operatingTemperature],
                                    'postWeldHeatTreatment':[postWeldHeatTreatment],
                                    'hydrogenContains':[hydrogenContains],
                                    'MBHW':[MBHW],
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