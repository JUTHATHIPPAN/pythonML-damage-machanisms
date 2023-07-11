import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from models.predictTest import DamagePredictionRequest, DamagePredictionResponse
from fastapi import APIRouter
from services import damageModel1, damageModel2, damageModel3, damageModel4, damageModel5, damageModel6, damageModel7, damageModel8, damageModel9, damageModel10, damageModel11, damageModel12, damageModel13, damageModel14, damageModel15, damageModel16, damageModel17, damageModel18, damageModel19, damageModel20, damageModel21, damageModel22, damageModel23, damageModel24

router = APIRouter()
@router.post("/")
def process_request(request: DamagePredictionRequest):
    damage_info=[]
    
    material = request.material
    operatingPressure = request.operatingPressure
    operatingTemperature = request.operatingTemperature
    designPressure = request.designPressure
    designTemperature = request.designTemperature
    insulation = request.insulation
    amineContains = request.amineContains
    postWeldHeatTreatment = request.postWeldHeatTreatment
    hydrogenContains = request.hydrogenContains
    ultimateTensileStrength = request.ultimateTensileStrength
    operatingHydrogenPartialPressure = request.operatingHydrogenPartialPressure
    h2sContains = request.h2sContains
    TAN = request.TAN
    fluidPhase = request.fluidPhase
    sulfurContains = request.sulfurContains
    modelFluid = request.modelFluid
    exposureToErosion = request.exposureToErosion
    injectionPoint = request.injectionPoint
    solidParticle_droplets = request.solidParticle_droplets
    MDMT = request.MDMT
    MDMT_MAT = request.MDMT_MAT
    waterContains = request.waterContains
    externalEnvironment = request.externalEnvironment
    pH = request.pH
    junctionOfDissimilarMetals = request.junctionOfDissimilarMetals
    nickel_basedAlloy = request.nickel_basedAlloy
    oxygenExist = request.oxygenExist
    MBHW = request.MBHW
    exposeToSoil = request.exposeToSoil
    
    # damage 1. Amine Corrosion
    if material is not None \
        and operatingTemperature is not None \
        and amineContains is not None:
        damage_info.append(damageModel1.predict(request))

    # damage 2. Amine Stress Corrosion Cracking
    if material is not None \
        and amineContains is not None \
        and postWeldHeatTreatment is not None:
        damage_info.append(damageModel2.predict(request))
        
    # damage 3. Corrosion Under Insulation
    if material is not None \
        and operatingTemperature is not None \
        and insulation is not None:
        damage_info.append(damageModel3.predict(request))
        
    # damage 4. Creep and Stress Rupture
    if (material == 1
        and designTemperature is not None 
        and ultimateTensileStrength is not None
        ) \
        or ((material == 2 or material == 3 or material == 4 or material == 5)
            and designTemperature is not None
        ) :
        damage_info.append(damageModel4.predict(request))
    
    # damage 5. High-temperature Hydrogen Attack
    if material is not None \
        and operatingTemperature is not None \
        and postWeldHeatTreatment is not None \
        and hydrogenContains is not None \
        and operatingHydrogenPartialPressure is not None:
        damage_info.append(damageModel5.predict(request))
        
    # damage 6. Atmospheric Corrosion
    if material is not None \
        and operatingTemperature is not None \
        and insulation is not None:
        damage_info.append(damageModel6.predict(request))
        
    # damage 7. Erosion/Erosion-Corrosion
    if exposureToErosion is not None \
        and injectionPoint is not None \
        and solidParticle_droplets is not None :
        damage_info.append(damageModel7.predict(request))
    
    # damage 8. High-temperature H2/H2S Corrosion
    if material is not None \
        and operatingTemperature is not None \
        and hydrogenContains is not None \
        and h2sContains is not None:
        damage_info.append(damageModel8.predict(request))
        
    # damage 9. Naphthenic Acid Corrosion
    if operatingTemperature is not None \
        and TAN is not None \
        and fluidPhase is not None \
        and sulfurContains is not None\
        and modelFluid is not None:
        damage_info.append(damageModel9.predict(request))
        
    # damage 10. Anhydrous Ammonia Stress Corrosion Cracking
    if material is not None \
        and operatingTemperature is not None \
        and postWeldHeatTreatment is not None \
        and ultimateTensileStrength is not None\
        and modelFluid is not None\
        and waterContains is not None:
        damage_info.append(damageModel10.predict(request))
        
    # damage 11. Brittle Fracture
    if material is not None \
        and operatingTemperature is not None \
        and MDMT is not None \
        and MDMT_MAT is not None:
        damage_info.append(damageModel11.predict(request))
        
    # damage 12. Caustic Corrosion
    if (
            material in [1,2] 
            and modelFluid is not None
        ) \
    or (
            material in [3,4,5] 
            and operatingTemperature is not None 
            and modelFluid is not None
        ):
        damage_info.append(damageModel12.predict(request))
    
    # damage 13. Caustic Stress Corrosion Cracking
    if postWeldHeatTreatment is not None \
        and modelFluid is not None:
        damage_info.append(damageModel13.predict(request))
    
    # damage 14. Chloride Stress Corrosion Cracking
    if (material is not None and externalEnvironment is not None) \
        or (material is not None \
        and operatingTemperature is not None \
        and postWeldHeatTreatment is not None \
        and modelFluid is not None \
        and waterContains is not None \
        and pH is not None):
        damage_info.append(damageModel14.predict(request))
        
    # damage 15. Galvanic Corrosion
    if junctionOfDissimilarMetals is not None :
        damage_info.append(damageModel15.predict(request))
    
    # damage 16. Hydrochloric Acid Corrosion
    if modelFluid is not None \
        and pH is not None:
        damage_info.append(damageModel16.predict(request))
        
    # damage 17. Hydrofluoric Acid Corrosion
    if modelFluid is not None \
        and waterContains is not None:
        damage_info.append(damageModel17.predict(request))
    
    # damage 18. Hydrofluoric Acid Stress Corrosion Cracking of Nickel Alloys
    if postWeldHeatTreatment is not None \
        and modelFluid is not None \
        and nickel_basedAlloy is not None \
        and oxygenExist is not None:
        damage_info.append(damageModel18.predict(request))
    
    # damage 19. Hydrogen Stress Cracking in Hydrofluoric Acid
    if material is not None \
        and postWeldHeatTreatment is not None \
        and modelFluid is not None \
        and MBHW is not None:
        damage_info.append(damageModel19.predict(request))
    
    # damage 20. Flue Gas Dew Point Corrosion
    if material is not None \
        and operatingTemperature is not None \
        and modelFluid is not None :
        damage_info.append(damageModel20.predict(request))
    
    # damage 21. Hydrogen Embrittlement
    if operatingTemperature is not None \
        and postWeldHeatTreatment is not None \
        and hydrogenContains is not None \
        and MBHW is not None :
        damage_info.append(damageModel21.predict(request))
    
    # damage 22. Oxidation
    if (
            material in [1,2] 
            and operatingTemperature is not None
        ) \
    or (
            material in [3,4,5] 
            and operatingTemperature is not None 
            and oxygenExist is not None
        ):
        damage_info.append(damageModel22.predict(request))
    
    # damage 23. Soil Corrosion
    if material is not None \
        and exposeToSoil is not None :
        damage_info.append(damageModel23.predict(request))
    
    # damage 24. Sour Water Corrosion (Acidic)
    if material is not None \
        and h2sContains is not None \
        and pH is not None :
        damage_info.append(damageModel24.predict(request))
        
    return DamagePredictionResponse(damage_info=damage_info)