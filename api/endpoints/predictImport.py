import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, UploadFile, File
from models.predictImport import FileUpload, DamagePrediction
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from core.config import cursor
import math
from services import damageModel1, damageModel2, damageModel3, damageModel4, damageModel5, damageModel6, damageModel7, damageModel8, damageModel9, damageModel10, damageModel11, damageModel12, damageModel13, damageModel14, damageModel15, damageModel16, damageModel17, damageModel18, damageModel19, damageModel20, damageModel21, damageModel22, damageModel23, damageModel24

MatModel = joblib.load('models_ml/material/MatModel.joblib')
# Initialize the CountVectorizer
vectorizerMatModel = CountVectorizer()
vectorizerMatModel.vocabulary_ = joblib.load('models_ml/material/VocabMatModel.joblib')

# Function to search material by grade
def search_material_by_grade(material_id):
    # Execute a SELECT query
    cursor.execute("SELECT * FROM [dbo].[tb_basic_material] WHERE id = %s", (material_id,))
    # Fetch all the rows returned by the query
    rows = cursor.fetchall()
    # print(rows)
    return rows

def predict_marerial_grade(id):
    input_string = id
    characters_to_remove = [" ", "/", "-", "."]
    for char in characters_to_remove:
        input_string = input_string.replace(char, "")

    # Transform the input string using the vectorizer
    input_vector = vectorizerMatModel.transform([input_string])
    prediction = MatModel.predict(input_vector)
    material_info = search_material_by_grade(str(prediction[0]))

    if material_info:
        response = {
                        'Material ID': material_info[0][6],
                        'Material Grade': material_info[0][1],
                        'Base Material': material_info[0][4],
                        'Composition': material_info[0][3],
                        'Ultimate Tensile Strength': material_info[0][5],
                    }
        return response

router = APIRouter()
@router.post("/")
async def upload_file(files: UploadFile = File(...)):
    
    content = files.file.read()
    df = pd.read_excel(content, sheet_name="Record", header=3)

    column_names = [
                        'Equipment ID',
                        'Material Grade',
                        'Operating Pressure [Barg]',
                        'Operating Temperature [°C]',
                        'Design Pressure [Barg]',
                        'Design Temperature [°C]',
                        'Insulation',
                        'Amine Contains [wt%]',
                        'Post Weld Heat Treatment',
                        'Hydrogen Contains [wt%]',
                        'Ultimate Tensile Strength [ksi]',
                        'Operating Hydrogen Partial Pressure [Barg]',
                        'H2S Contains [%Mole]',
                        'TAN [mg/g]',
                        'Fluid Phase',
                        'Sulfur Contains [wt%]',
                        'Model Fluid',
                        'Exposure to Erosion',
                        'Injection Point',
                        'Solid Particle / Droplets',
                        'Minimum Design Metal Temperature (MDMT) [°C]',
                        'The component is operate at or below the MDMT or MAT under upset conditions.',
                        'Water contains in process conditions [%wt]',
                        'External Environment',
                        'pH',
                        'Junction of Dissimilar Metals',
                        'Nickel-based Alloy',
                        'Oxygen exist in operating/shutdown condition or in purge gases (nitrogen, fuel gas)',
                        'Maximum Brinell Hardness of Weld',
                        'Expose to Soil'
                    ]

    rows = []
    # Loop through each row of the DataFrame
    for _, row in df.iterrows():
        collect = {}
        for column in column_names:
            value = row[column]
            collect[column] = value
        # print(collect['Material Grade'])
        matInfo = predict_marerial_grade(collect['Material Grade'])
        # print(matInfo['Ultimate Tensile Strength'])
        if math.isnan(collect['Ultimate Tensile Strength [ksi]']):
            collect['Ultimate Tensile Strength [ksi]'] = matInfo['Ultimate Tensile Strength']
        collect['Material'] = matInfo['Base Material']
        collect['Material ID'] = matInfo['Material ID']
        damages = predict_damage(
                    DamagePrediction(
                        material = collect['Material ID'],
                        operatingPressure = collect['Operating Pressure [Barg]'],
                        operatingTemperature = collect['Operating Temperature [°C]'],
                        designPressure = collect['Design Pressure [Barg]'],
                        designTemperature = collect['Design Temperature [°C]'],
                        insulation = yesNoCheck(collect['Insulation']),
                        amineContains = collect['Amine Contains [wt%]'],
                        postWeldHeatTreatment = yesNoCheck(collect['Post Weld Heat Treatment']),
                        hydrogenContains = collect['Hydrogen Contains [wt%]'],
                        ultimateTensileStrength = collect['Ultimate Tensile Strength [ksi]'],
                        operatingHydrogenPartialPressure = collect['Operating Hydrogen Partial Pressure [Barg]'],
                        h2sContains = collect['H2S Contains [%Mole]'],
                        TAN = collect['TAN [mg/g]'],
                        fluidPhase = fluidPhaseCheck(collect['Fluid Phase']),
                        sulfurContains = collect['Sulfur Contains [wt%]'],
                        modelFluid = modelFluidCheck(collect['Model Fluid']),
                        exposureToErosion = yesNoCheck(collect['Exposure to Erosion']),
                        injectionPoint = yesNoCheck(collect['Injection Point']),
                        solidParticle_droplets = yesNoCheck(collect['Solid Particle / Droplets']),
                        MDMT = collect['Minimum Design Metal Temperature (MDMT) [°C]'],
                        MDMT_MAT = yesNoCheck(collect['The component is operate at or below the MDMT or MAT under upset conditions.']),
                        waterContains = collect['Water contains in process conditions [%wt]'],
                        externalEnvironment = externalEnvironmentCheck(collect['External Environment']),
                        pH = collect['pH'],
                        junctionOfDissimilarMetals = yesNoCheck(collect['Junction of Dissimilar Metals']),
                        nickel_basedAlloy = yesNoCheck(collect['Nickel-based Alloy']),
                        oxygenExist = collect['Oxygen exist in operating/shutdown condition or in purge gases (nitrogen, fuel gas)'],
                        MBHW = collect['Maximum Brinell Hardness of Weld'],
                        exposeToSoil = yesNoCheck(collect['Expose to Soil']),
                        )
                    )
        # print(collect['Equipment ID'],damage)
        for i in damages: 
            damge = i['damage']
            print(collect['Equipment ID'],damge)
        # print(collect['Equipment ID'],damage[0]['damage'])
        # collect['Result Damage Mechanisms'] = damage[0]['damage']
        # print(collect, damage[0]['damage'])
        rows.append(collect)

    # Convert rows to JSON-compliant format
    rows_json = json.dumps(rows)
    # print(rows)
    return rows_json
    
def predict_damage(feature):
    damage_info=[]
    material = feature.material
    operatingPressure = feature.operatingPressure
    operatingTemperature = feature.operatingTemperature
    designPressure = feature.designPressure
    designTemperature = feature.designTemperature
    insulation = feature.insulation
    amineContains = feature.amineContains
    postWeldHeatTreatment = feature.postWeldHeatTreatment
    hydrogenContains = feature.hydrogenContains
    ultimateTensileStrength = feature.ultimateTensileStrength
    operatingHydrogenPartialPressure = feature.operatingHydrogenPartialPressure
    h2sContains = feature.h2sContains
    TAN = feature.TAN
    fluidPhase = feature.fluidPhase
    sulfurContains = feature.sulfurContains
    modelFluid = feature.modelFluid
    exposureToErosion = feature.exposureToErosion
    injectionPoint = feature.injectionPoint
    solidParticle_droplets = feature.solidParticle_droplets
    MDMT = feature.MDMT
    MDMT_MAT = feature.MDMT_MAT
    waterContains = feature.waterContains
    externalEnvironment = feature.externalEnvironment
    pH = feature.pH
    junctionOfDissimilarMetals = feature.junctionOfDissimilarMetals
    nickel_basedAlloy = feature.nickel_basedAlloy
    oxygenExist = feature.oxygenExist
    MBHW = feature.MBHW
    exposeToSoil = feature.exposeToSoil
    # damage 1. Amine Corrosion
    # print(math.isnan(operatingTemperature))
    # print(ultimateTensileStrength)
    
    if material is not None \
        and not math.isnan(operatingTemperature) \
        and not math.isnan(amineContains):
        damage_info.append(damageModel1.predict(feature))

    # damage 2. Amine Stress Corrosion Cracking
    if material is not None \
        and not math.isnan(amineContains) \
        and postWeldHeatTreatment is not None:
        damage_info.append(damageModel2.predict(feature))
        
    # damage 3. Corrosion Under Insulation
    if material is not None \
        and not math.isnan(operatingTemperature) \
        and insulation is not None:
        damage_info.append(damageModel3.predict(feature))
        
    # damage 4. Creep and Stress Rupture
    if (material == 1
        and not math.isnan(designTemperature) 
        and not math.isnan(ultimateTensileStrength)
        ) \
        or ((material == 2 or material == 3 or material == 4 or material == 5)
            and math.isnan(designTemperature)
        ) :
        damage_info.append(damageModel4.predict(feature))
    
    # damage 5. High-temperature Hydrogen Attack
    if material is not None \
        and not math.isnan(operatingTemperature)\
        and postWeldHeatTreatment is not None \
        and not math.isnan(hydrogenContains) \
        and not math.isnan(operatingHydrogenPartialPressure):
        damage_info.append(damageModel5.predict(feature))
        
    # damage 6. Atmospheric Corrosion
    if material is not None \
        and not math.isnan(operatingTemperature) \
        and insulation is not None:
        damage_info.append(damageModel6.predict(feature))
        
    # damage 7. Erosion/Erosion-Corrosion
    if exposureToErosion is not None \
        and injectionPoint is not None \
        and solidParticle_droplets is not None :
        damage_info.append(damageModel7.predict(feature))
    
    # damage 8. High-temperature H2/H2S Corrosion
    if material is not None \
        and not math.isnan(operatingTemperature) \
        and not math.isnan(hydrogenContains) \
        and not math.isnan(h2sContains):
        damage_info.append(damageModel8.predict(feature))
        
    # damage 9. Naphthenic Acid Corrosion
    if not math.isnan(operatingTemperature) \
        and not math.isnan(TAN) \
        and fluidPhase is not None \
        and not math.isnan(sulfurContains)\
        and modelFluid is not None:
        damage_info.append(damageModel9.predict(feature))
        
    # damage 10. Anhydrous Ammonia Stress Corrosion Cracking
    if material is not None \
        and not math.isnan(operatingTemperature) \
        and postWeldHeatTreatment is not None \
        and ultimateTensileStrength is not None\
        and modelFluid is not None\
        and not math.isnan(waterContains) :
        damage_info.append(damageModel10.predict(feature))
        
    # damage 11. Brittle Fracture
    if material is not None \
        and not math.isnan(operatingTemperature) \
        and not math.isnan(MDMT) \
        and MDMT_MAT is not None:
        damage_info.append(damageModel11.predict(feature))
        
    # damage 12. Caustic Corrosion
    if (
            material in [1,2] 
            and modelFluid is not None
        ) \
    or (
            material in [3,4,5] 
            and not math.isnan(operatingTemperature) 
            and modelFluid is not None
        ):
        damage_info.append(damageModel12.predict(feature))
    
    # damage 13. Caustic Stress Corrosion Cracking
    if postWeldHeatTreatment is not None \
        and modelFluid is not None:
        damage_info.append(damageModel13.predict(feature))
    
    # print(postWeldHeatTreatment is not None)
    # damage 14. Chloride Stress Corrosion Cracking
    if (
            material is not None 
            and externalEnvironment is not None
        ) \
        or (material is not None \
        and not math.isnan(operatingTemperature) \
        and postWeldHeatTreatment is not None \
        and modelFluid is not None \
        and not math.isnan(waterContains) \
        and not math.isnan(pH)) :
        damage_info.append(damageModel14.predict(feature))
        
    # damage 15. Galvanic Corrosion
    if junctionOfDissimilarMetals is not None :
        damage_info.append(damageModel15.predict(feature))
    
    # damage 16. Hydrochloric Acid Corrosion
    if modelFluid is not None \
        and not math.isnan(pH):
        damage_info.append(damageModel16.predict(feature))
        
    # damage 17. Hydrofluoric Acid Corrosion
    if modelFluid is not None \
        and not math.isnan(waterContains):
        damage_info.append(damageModel17.predict(feature))
    
    # damage 18. Hydrofluoric Acid Stress Corrosion Cracking of Nickel Alloys
    if postWeldHeatTreatment is not None \
        and modelFluid is not None \
        and nickel_basedAlloy is not None \
        and oxygenExist is not None:
        damage_info.append(damageModel18.predict(feature))
    
    # damage 19. Hydrogen Stress Cracking in Hydrofluoric Acid
    if material is not None \
        and postWeldHeatTreatment is not None \
        and modelFluid is not None \
        and not math.isnan(MBHW):
        damage_info.append(damageModel19.predict(feature))
    
    # damage 20. Flue Gas Dew Point Corrosion
    if material is not None \
        and not math.isnan(operatingTemperature) \
        and modelFluid is not None :
        damage_info.append(damageModel20.predict(feature))
    
    # damage 21. Hydrogen Embrittlement
    if not math.isnan(operatingTemperature) \
        and postWeldHeatTreatment is not None \
        and not math.isnan(hydrogenContains) \
        and not math.isnan(MBHW):
        damage_info.append(damageModel21.predict(feature))
    
    # damage 22. Oxidation
    if (
            material in [1,2] 
            and not math.isnan(operatingTemperature)
        ) \
    or (
            material in [3,4,5] 
            and not math.isnan(operatingTemperature) 
            and not math.isnan(oxygenExist)
        ):
        damage_info.append(damageModel22.predict(feature))
    
    # damage 23. Soil Corrosion
    if material is not None \
        and exposeToSoil is not None :
        damage_info.append(damageModel23.predict(feature))
    
    # damage 24. Sour Water Corrosion (Acidic)
    if material is not None \
        and not math.isnan(h2sContains) \
        and not math.isnan(pH):
        damage_info.append(damageModel24.predict(feature))
    
    
    damage_predictions = [damage for damage in damage_info if damage['damage'] != 'General Corrosion']
    if  isinstance(damage_predictions, list) and len(damage_predictions) > 0: return damage_predictions
    else: 
        return [{'model': 'all','damage': 'General Corrosion', 'proba_percent': 100}]
         

def yesNoCheck(feature):
    if feature == "Yes":
        return 1
    elif feature == "No":
        return 0
    else: return None
    
def fluidPhaseCheck(fluidPhase):
    if fluidPhase == "Liquid":
        return 1
    elif fluidPhase == "Gas":
        return 2
    else: return None
    
def externalEnvironmentCheck(externalEnvironment):
    if externalEnvironment == "Marine": return 1
    elif externalEnvironment == "Temperate": return 2
    elif externalEnvironment == "Arid/dry": return 3
    elif externalEnvironment == "Severe": return 4
    else: return None
    
def modelFluidCheck(modelFluid):
    if modelFluid == "Methane": return 1
    elif modelFluid == "Ethane": return 2
    elif modelFluid == "Ethylene": return 3
    elif modelFluid == "LNG": return 4
    elif modelFluid == "Fuel Gas": return 5
    elif modelFluid == "Propane": return 6
    elif modelFluid == "Butane": return 7
    elif modelFluid == "Isobutane": return 8
    elif modelFluid == "LPG": return 9
    elif modelFluid == "Pentane": return 10
    elif modelFluid == "Gasoline": return 11
    elif modelFluid == "Naphtha": return 12
    elif modelFluid == "Light Straight Run": return 13
    elif modelFluid == "Heptane": return 14
    elif modelFluid == "Diesel": return 15
    elif modelFluid == "Jet Fuel": return 16
    elif modelFluid == "Kerosene": return 17
    elif modelFluid == "Atmospheric Gas Oil": return 18
    elif modelFluid == "Gas Oil": return 19
    elif modelFluid == "Typical Crude": return 20
    elif modelFluid == "Residuum": return 21
    elif modelFluid == "Seal Oil": return 22
    elif modelFluid == "Lube Oil": return 23
    elif modelFluid == "Heavy Crude": return 24
    elif modelFluid == "Hydrogen": return 25
    elif modelFluid == "Hydrogen Sulfide (H2S)": return 26
    elif modelFluid == "Hydrogen Fluoride (HF)": return 27
    elif modelFluid == "Water": return 28
    elif modelFluid == "Steam": return 29
    elif modelFluid == "Acid, Caustic": return 30
    elif modelFluid == "Aromatics (Benzene, Toluene, Xylene, Cumene)": return 31
    elif modelFluid == "Aluminum Chloride (AlCl3)": return 32
    elif modelFluid == "Pyrophoric": return 33
    elif modelFluid == "Ammonia": return 34
    elif modelFluid == "Chlorine": return 35
    elif modelFluid == "Carbon Monoxide (CO)": return 36
    elif modelFluid == "Diethyl Ether": return 37
    elif modelFluid == "Hydrogen Chloride (HCL)": return 38
    elif modelFluid == "Nitric Acid": return 39
    elif modelFluid == "Nitrogen Dioxide": return 40
    elif modelFluid == "Phosgene": return 41
    elif modelFluid == "Toluene Diisocyanate": return 42
    elif modelFluid == "Methanol": return 43
    elif modelFluid == "Propylene Oxide": return 44
    elif modelFluid == "Styrene": return 45
    elif modelFluid == "Ethylene Glycol Monoethyl Ether Acetate": return 46
    elif modelFluid == "Ethylene Glycol Monoethyl Ether": return 47
    elif modelFluid == "Ethylene Glycol": return 48
    elif modelFluid == "Ethylene Oxide": return 49
    else: return None