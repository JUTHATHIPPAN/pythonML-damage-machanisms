import joblib
import pandas as pd
import numpy as np

modelNumber = 
damageName = ""

loaded_model = joblib.load('./models_ml/damages/model'+str(modelNumber)+'.joblib')

def predict(data: dict):
    material = data.material
    operatingPressure = data.operatingPressure
    operatingTemperature = data.operatingTemperature
    designPressure = data.designPressure
    designTemperature = data.designTemperature
    insulation = data.insulation
    amineContains = data.amineContains
    postWeldHeatTreatment = data.postWeldHeatTreatment
    hydrogenContains = data.hydrogenContains
    ultimateTensileStrength = data.ultimateTensileStrength
    operatingHydrogenPartialPressure = data.operatingHydrogenPartialPressure
    h2sContains = data.h2sContains
    TAN = data.TAN
    fluidPhase = data.fluidPhase
    sulfurContains = data.sulfurContains
    modelFluid = data.modelFluid
    exposureToErosion = data.exposureToErosion
    injectionPoint = data.injectionPoint
    solidParticle_droplets = data.solidParticle_droplets
    MDMT = data.MDMT
    MDMT_MAT = data.MDMT_MAT
    waterContains = data.waterContains
    externalEnvironment = data.externalEnvironment
    pH = data.pH
    junctionOfDissimilarMetals = data.junctionOfDissimilarMetals
    nickel_basedAlloy = data.nickel_basedAlloy
    oxygenExist = data.oxygenExist
    MBHW = data.MBHW
    exposeToSoil = data.exposeToSoil

    input_data = pd.DataFrame(
                                {
                                    'material': [material], 
                                    'operatingPressure': [operatingPressure],
                                    'operatingTemperature':[operatingTemperature],
                                    'designPressure':[designPressure],
                                    'designTemperature':[designTemperature],
                                    'insulation':[insulation],
                                    'amineContains':[amineContains],
                                    'postWeldHeatTreatment':[postWeldHeatTreatment],
                                    'hydrogenContains':[hydrogenContains],
                                    'ultimateTensileStrength':[ultimateTensileStrength],
                                    'operatingHydrogenPartialPressure':[operatingHydrogenPartialPressure],
                                    'h2sContains':[h2sContains],
                                    'TAN':[TAN],
                                    'fluidPhase':[fluidPhase],
                                    'sulfurContains':[sulfurContains],
                                    'modelFluid':[modelFluid],
                                    'exposureToErosion':[exposureToErosion],
                                    'injectionPoint':[injectionPoint],
                                    'solidParticle_droplets':[solidParticle_droplets],
                                    'MDMT':[MDMT],
                                    'MDMT_MAT':[MDMT_MAT],
                                    'waterContains':[waterContains],
                                    'externalEnvironment':[externalEnvironment],
                                    'pH':[pH],
                                    'junctionOfDissimilarMetals':[junctionOfDissimilarMetals],
                                    'nickel_basedAlloy':[nickel_basedAlloy],
                                    'oxygenExist':[oxygenExist],
                                    'MBHW':[MBHW],
                                    'exposeToSoil':[exposeToSoil]
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