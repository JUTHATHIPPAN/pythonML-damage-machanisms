from pydantic import BaseModel
from typing import Optional

class DamagePredictionRequest(BaseModel):
    material: Optional[int] = None
    operatingPressure: Optional[float] = None
    operatingTemperature: Optional[float] = None
    designPressure: Optional[float] = None
    designTemperature: Optional[float] = None
    insulation: Optional[int] = None
    amineContains: Optional[float] = None
    postWeldHeatTreatment: Optional[int] = None
    hydrogenContains: Optional[float] = None
    ultimateTensileStrength: Optional[float] = None
    operatingHydrogenPartialPressure: Optional[float] = None
    h2sContains: Optional[float] = None
    TAN: Optional[float] = None
    fluidPhase: Optional[int] = None
    sulfurContains: Optional[float] = None
    modelFluid: Optional[int] = None
    exposureToErosion: Optional[int] = None
    injectionPoint: Optional[int] = None
    solidParticle_droplets: Optional[int] = None
    MDMT: Optional[int] = None
    MDMT_MAT: Optional[int] = None
    waterContains: Optional[float] = None
    externalEnvironment: Optional[int] = None
    pH: Optional[float] = None
    junctionOfDissimilarMetals: Optional[int] = None
    nickel_basedAlloy: Optional[int] = None
    oxygenExist: Optional[int] = None
    MBHW: Optional[float] = None
    exposeToSoil: Optional[int] = None
    

class DamagePredictionResponse(BaseModel):
    damage_info: Optional[list] = None