import joblib
import pandas as pd
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.predictImport import FileUpload, projectListRes, equipmentRes, equipmentReq
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from core.config import cursor
import math
from services import damageModel1, damageModel2, damageModel3, damageModel4, damageModel5, damageModel6, damageModel7, damageModel8, damageModel9, damageModel10, damageModel11, damageModel12, damageModel13, damageModel14, damageModel15, damageModel16, damageModel17, damageModel18, damageModel19, damageModel20, damageModel21, damageModel22, damageModel23, damageModel24
from typing import List

MatModel = joblib.load('models_ml/material/MatModel.joblib')
# Initialize the CountVectorizer
vectorizerMatModel = CountVectorizer()
vectorizerMatModel.vocabulary_ = joblib.load('models_ml/material/VocabMatModel.joblib')

router = APIRouter()

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
    # print(math.isnan(operatingTemperature))
    # print(ultimateTensileStrength)
    
    # damage 1. Amine Corrosion
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
    
def saveToTb_equipment(collect):
    try:
        # Execute SQL query
        queryTb_equipment = """
                                INSERT INTO [ML_damage].[dbo].[tb_equipment] 
                                    (
                                        equipment_id,
                                        project_id,
                                        IsActive
                                    ) 
                                VALUES 
                                    (
                                        %s,
                                        %s,
                                        %s
                                    )
                            """
        cursor.execute(queryTb_equipment, (collect['Equipment ID'], collect['project ID'], 1))
        cursor.connection.commit()
        inserted_id = cursor.lastrowid
        
        return inserted_id
    finally:
        pass

def saveToTb_equipment_info(collect):
    try:
        # Execute SQL query
        query = """
            INSERT INTO [ML_damage].[dbo].[tb_equipment_info] 
            (
                equipment_id, 
                materialGrade_id, 
                operatingTemperature, 
                designTemperature, 
                operatingPressure, 
                designPressure, 
                insulation,
                amineContains,
                postWeldHeatTreatment,
                hydrogenContains,
                ultimateTensileStrength,
                operatingHydrogenPartialPressure,
                h2sContains,
                TAN,
                fluidPhase,
                sulfurContains,
                modelFluid,
                exposureToErosion,
                injectionPoint,
                solidParticle_droplets,
                MDMT,
                MDMT_MAT,
                waterContains,
                externalEnvironment,
                pH,
                junctionOfDissimilarMetals,
                nickel_basedAlloy,
                oxygenExist,
                MBHW,
                exposeToSoil,
                IsActive
            ) 
            VALUES 
            (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                1
            )
        """
        cursor.execute(query, (
                                collect['equipment_id'], #Equipment ID of tb_equipment_info
                                collect['materialGrade_id'],
                                collect['operatingTemperature'],
                                collect['designTemperature'],
                                collect['operatingPressure'],
                                collect['designPressure'],
                                collect['insulation'],
                                collect['amineContains'],
                                collect['postWeldHeatTreatment'],
                                collect['hydrogenContains'],
                                collect['ultimateTensileStrength'],
                                collect['operatingHydrogenPartialPressure'],
                                collect['h2sContains'],
                                collect['TAN'],
                                collect['fluidPhase'],
                                collect['sulfurContains'],
                                collect['modelFluid'],
                                collect['exposureToErosion'],
                                collect['injectionPoint'],
                                collect['solidParticle_droplets'],
                                collect['MDMT'],
                                collect['MDMT_MAT'],
                                collect['waterContains'],
                                collect['externalEnvironment'],
                                collect['pH'],
                                collect['junctionOfDissimilarMetals'],
                                collect['nickel_basedAlloy'],
                                collect['oxygenExist'],
                                collect['MBHW'],
                                collect['exposeToSoil'],
                            )
                       )

        # Perform commit on the connection object
        cursor.connection.commit()

        inserted_id = cursor.lastrowid

        return inserted_id
    finally:
        pass

def saveToTb_result(rowsID, damage, prob):
    try:
        # Execute SQL query
        queryTb_result = """
                                INSERT INTO [ML_damage].[dbo].[tb_result] 
                                    (
                                        equipment_Info_id,
                                        damage,
                                        probability,
                                        IsActive
                                    ) 
                                VALUES 
                                (
                                    %s,
                                    %s,
                                    %s,
                                    1
                                )
                            """
        cursor.execute(queryTb_result, (rowsID, damage, prob))
        cursor.connection.commit()
        return 
    finally:
        pass

def handle_nan_values(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

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
                        'id': material_info[0][0],
                    }
        return response
    
@router.post("/")
async def upload_file(files: UploadFile = File(...), projectID: int = None):
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

        # Execute SQL query (tb_equipment)
        value_tb_equipment = {}
        value_tb_equipment['Equipment ID'] = collect['Equipment ID'] #Equipment ID of tb_equipment
        value_tb_equipment['project ID'] = projectID
        value_tb_equipment = {key: handle_nan_values(value) for key, value in value_tb_equipment.items()}
        inserted_tb_equipment_id = saveToTb_equipment(value_tb_equipment)
        
        # Execute SQL query (tb_equipment_info)
        value_tb_equipment_info = {}
        value_tb_equipment_info['equipment_id'] = inserted_tb_equipment_id #Equipment ID of tb_equipment_info
        value_tb_equipment_info['materialGrade_id'] = matInfo['Material ID']
        value_tb_equipment_info['operatingTemperature'] = collect['Operating Temperature [°C]']
        value_tb_equipment_info['designTemperature'] = collect['Design Temperature [°C]']
        value_tb_equipment_info['operatingPressure'] = collect['Operating Pressure [Barg]']
        value_tb_equipment_info['designPressure'] = collect['Design Pressure [Barg]']
        value_tb_equipment_info['insulation'] = yesNoCheck(collect['Insulation']),
        value_tb_equipment_info['amineContains'] = collect['Amine Contains [wt%]']
        value_tb_equipment_info['postWeldHeatTreatment'] = yesNoCheck(collect['Post Weld Heat Treatment']),
        value_tb_equipment_info['hydrogenContains'] = collect['Hydrogen Contains [wt%]']
        value_tb_equipment_info['ultimateTensileStrength'] = collect['Ultimate Tensile Strength [ksi]']
        value_tb_equipment_info['operatingHydrogenPartialPressure'] = collect['Operating Hydrogen Partial Pressure [Barg]']
        value_tb_equipment_info['h2sContains'] = collect['H2S Contains [%Mole]']
        value_tb_equipment_info['TAN'] = collect['TAN [mg/g]']
        value_tb_equipment_info['fluidPhase'] = fluidPhaseCheck(collect['Fluid Phase']),
        value_tb_equipment_info['sulfurContains'] = collect['Sulfur Contains [wt%]']
        value_tb_equipment_info['modelFluid'] = modelFluidCheck(collect['Model Fluid']),
        value_tb_equipment_info['exposureToErosion'] = yesNoCheck(collect['Exposure to Erosion']),
        value_tb_equipment_info['injectionPoint'] = yesNoCheck(collect['Injection Point']),
        value_tb_equipment_info['solidParticle_droplets'] = yesNoCheck(collect['Solid Particle / Droplets']),
        value_tb_equipment_info['MDMT'] = collect['Minimum Design Metal Temperature (MDMT) [°C]']
        value_tb_equipment_info['MDMT_MAT'] = yesNoCheck(collect['The component is operate at or below the MDMT or MAT under upset conditions.']),
        value_tb_equipment_info['waterContains'] = collect['Water contains in process conditions [%wt]']
        value_tb_equipment_info['externalEnvironment'] = externalEnvironmentCheck(collect['External Environment']),
        value_tb_equipment_info['pH'] = collect['pH']
        value_tb_equipment_info['junctionOfDissimilarMetals'] = yesNoCheck(collect['Junction of Dissimilar Metals']),
        value_tb_equipment_info['nickel_basedAlloy'] = yesNoCheck(collect['Nickel-based Alloy']),
        value_tb_equipment_info['oxygenExist'] = collect['Oxygen exist in operating/shutdown condition or in purge gases (nitrogen, fuel gas)']
        value_tb_equipment_info['MBHW'] = collect['Maximum Brinell Hardness of Weld']
        value_tb_equipment_info['exposeToSoil'] = yesNoCheck(collect['Expose to Soil']),
        value_tb_equipment_info = {key: handle_nan_values(value) for key, value in value_tb_equipment_info.items()}
        inserted_tb_equipment_info_id = saveToTb_equipment_info(value_tb_equipment_info)
        
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
        for i in damages:
            saveToTb_result(
                                inserted_tb_equipment_info_id,
                                i['damage'],
                                i['proba_percent']
                            )
            
    return HTTPException(status_code=200, detail='suscess')

@router.get("/project", response_model=List[projectListRes])
async def projectList():
    try:
        # Execute SQL query to fetch user details
        query = """
                    SELECT id, project_id, project_name, remark, added_date, updated_date
                    FROM [ML_damage].[dbo].[tb_project]
                    WHERE isActive = 1;
                """
        cursor.execute(query)
        result = cursor.fetchall()

        projects = []
        for row in result:
            project = projectListRes(
                id = row[0],
                project_id=row[1],
                project_name=row[2],
                remark=row[3],
                added_date=row[4],
                updated_date=row[5]
            )
            projects.append(project)

        # Return the list of projects
        return projects

    except Exception as e:
        print('Error:', str(e))
        raise HTTPException(status_code=500, detail='Internal Server Error')

@router.post("/equipment", response_model=List[equipmentRes])
async def equipmentList(Req:equipmentReq):
    try:
        # Execute SQL query to fetch user details
        query = """
                    SELECT  e.id as id,
                            e.equipment_id as equipment_id,
                            einfo.id as equipment_info_id,
                            einfo.materialGrade_id as materialGrade_id,
                            mat.grade_material as materialGrade,
                            einfo.operatingTemperature as operatingTemperature,
                            einfo.designTemperature as designTemperature,
                            einfo.operatingPressure as operatingPressure,
                            einfo.designPressure as designPressure,
                            yn_insu.name as insulation,
                            einfo.amineContains as amineContains,
                            yn_PWHT.name as postWeldHeatTreatment,
                            einfo.hydrogenContains as hydrogenContains,
                            einfo.ultimateTensileStrength as ultimateTensileStrength,
                            einfo.operatingHydrogenPartialPressure as operatingHydrogenPartialPressure,
                            einfo.h2sContains as h2sContains,
                            einfo.TAN as TAN,
                            fp.name as fluidPhase,
                            einfo.sulfurContains as sulfurContains,
                            mf.name as modelFluid,
                            yn_ete.name as exposureToErosion,
                            yn_inju.name as injectionPoint,
                            yn_spd.name as solidParticle_droplets,
                            einfo.MDMT as MDMT,
                            yn_MDMTMAT.name as MDMT_MAT,
                            einfo.waterContains as waterContains,
                            ee.name as externalEnvironment,
                            einfo.pH as pH,
                            jodm.name as junctionOfDissimilarMetals,
                            nba.name as nickel_basedAlloy,
                            einfo.oxygenExist as oxygenExist,
                            einfo.MBHW as MBHW,
                            ets.name as exposeToSoil,
                            e.added_date as added_date,
                            e.updated_date as updated_date

                    FROM [ML_damage].[dbo].[tb_project] as p
                    LEFT JOIN [ML_damage].[dbo].[tb_equipment] as e ON e.project_id = p.id
                    LEFT JOIN [ML_damage].[dbo].[tb_equipment_info] as einfo ON e.id = einfo.equipment_id
                    LEFT JOIN [ML_damage].[dbo].[tb_basic_material] as mat ON einfo.materialGrade_id = mat.id
                    LEFT JOIN [ML_damage].[dbo].[tb_fluid_phase] as fp ON einfo.fluidPhase = fp.id
                    LEFT JOIN [ML_damage].[dbo].[tb_model_fluid] as mf ON einfo.modelFluid = mf.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as yn_insu ON einfo.insulation = yn_insu.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as yn_PWHT ON einfo.insulation = yn_PWHT.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as yn_ete ON einfo.exposureToErosion = yn_ete.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as yn_inju ON einfo.injectionPoint = yn_inju.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as yn_spd ON einfo.solidParticle_droplets = yn_spd.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as yn_MDMTMAT ON einfo.MDMT_MAT = yn_MDMTMAT.id
                    LEFT JOIN [ML_damage].[dbo].[tb_external_environment] as ee ON einfo.externalEnvironment = ee.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as jodm ON einfo.junctionOfDissimilarMetals = jodm.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as nba ON einfo.nickel_basedAlloy = nba.id
                    LEFT JOIN [ML_damage].[dbo].[tb_yesNo] as ets ON einfo.exposeToSoil = ets.id
                    WHERE einfo.isActive = 1 AND p.id = %s;
                """
        cursor.execute(query,Req.project_id)
        result = cursor.fetchall()

        equipments = []
        for row in result:
            equipment = equipmentRes(
                id = row[0],
                equipment_id=row[1],
                equipment_info_id=row[2],
                materialGrade_id=row[3],
                materialGrade=row[4],
                operatingTemperature = row[5],
                designTemperature = row[6],
                operatingPressure = row[7],
                designPressure = row[8],
                insulation = row[9],
                amineContains = row[10],
                postWeldHeatTreatment = row[11],
                hydrogenContains = row[12],
                ultimateTensileStrength = row[13],
                operatingHydrogenPartialPressure = row[14],
                h2sContains = row[15],
                TAN = row[16],
                fluidPhase = row[17],
                sulfurContains = row[18],
                modelFluid = row[19],
                exposureToErosion = row[20],
                injectionPoint = row[21],
                solidParticle_droplets = row[22],
                MDMT = row[23],
                MDMT_MAT = row[24],
                waterContains = row[25],
                externalEnvironment = row[26],
                pH = row[27],
                junctionOfDissimilarMetals = row[28],
                nickel_basedAlloy = row[29],
                oxygenExist = row[30],
                MBHW = row[31],
                exposeToSoil = row[32],
                added_date = row[33],
                updated_date = row[34]
            )
            equipments.append(equipment)

        # Return the list of projects
        return equipments

    except Exception as e:
        print('Error:', str(e))
        raise HTTPException(status_code=500, detail='Internal Server Error')

