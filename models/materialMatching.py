from pydantic import BaseModel

class MaterailGradePredictionRequest(BaseModel):
    grade: str

class MaterailGradePredictionResponse(BaseModel):
    Grade_Material: str
    Base_Material: str
    Composition: str
    Base_Material_Abbreviation: str
    Ultimate_Tensile_Strength_ksi: int
    Material_ID: int
