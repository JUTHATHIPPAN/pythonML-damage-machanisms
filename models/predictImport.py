from pydantic import BaseModel
from fastapi import UploadFile, File
from typing import Optional
from datetime import date

class FileUpload(BaseModel):
    projectID: int
    files: bytes
    
class projectListRes(BaseModel):
    id: Optional[str] = None
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    remark: Optional[str] = None
    added_date: Optional[date] = None
    updated_date: Optional[date] = None

class equipmentReq(BaseModel):
    project_id: Optional[int] = None
    
class equipmentRes(BaseModel):
    id: Optional[int] = None #id of tb_equipment
    equipment_id: Optional[str] = None #name of equipment
    equipment_info_id: Optional[int] = None #id of tb_equipment_info
    materialGrade_id: Optional[float] = None
    materialGrade: Optional[str] = None
    operatingTemperature: Optional[float] = None
    designTemperature: Optional[float] = None
    operatingPressure: Optional[float] = None
    designPressure: Optional[float] = None
    insulation: Optional[str] = None
    amineContains: Optional[float] = None
    postWeldHeatTreatment: Optional[str] = None
    hydrogenContains: Optional[float] = None
    ultimateTensileStrength: Optional[float] = None
    operatingHydrogenPartialPressure: Optional[float] = None
    h2sContains: Optional[float] = None
    TAN: Optional[float] = None
    fluidPhase: Optional[str] = None
    sulfurContains: Optional[float] = None
    modelFluid: Optional[str] = None
    exposureToErosion: Optional[str] = None
    injectionPoint: Optional[str] = None
    solidParticle_droplets: Optional[str] = None
    MDMT: Optional[float] = None
    MDMT_MAT: Optional[str] = None
    waterContains: Optional[float] = None
    externalEnvironment: Optional[str] = None
    pH: Optional[float] = None
    junctionOfDissimilarMetals: Optional[str] = None
    nickel_basedAlloy: Optional[str] = None
    oxygenExist: Optional[float] = None
    MBHW: Optional[float] = None
    exposeToSoil: Optional[str] = None
    added_date:Optional[date] = None
    updated_date:Optional[date] = None