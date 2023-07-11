from pydantic import BaseModel

class DamagePredictionRequest(BaseModel):
    material: float
    operating_temperature: float
    amine_contains: float

class DamagePredictionResponse(BaseModel):
    damage: str
    probability: int