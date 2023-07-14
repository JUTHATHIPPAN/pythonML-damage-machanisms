import joblib
import pandas as pd
import numpy as np

modelNumber = 19
damageName = "Hydrogen Stress Cracking in Hydrofluoric Acid"

loaded_model = joblib.load('./models_ml/damages/model'+str(modelNumber)+'.joblib')

def predict(data: dict):
    material = data.material
    postWeldHeatTreatment = data.postWeldHeatTreatment
    modelFluid = data.modelFluid
    MBHW = data.MBHW
    input_data = pd.DataFrame(
                                {
                                    'material': [material], 
                                    'postWeldHeatTreatment':[postWeldHeatTreatment],
                                    'modelFluid':[modelFluid],
                                    'MBHW':[MBHW]
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