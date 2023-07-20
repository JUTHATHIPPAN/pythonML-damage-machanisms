from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from api.endpoints.predict import router as predict_router
from api.endpoints.predictTest import router as predictTest_router
from api.endpoints.materialMatching import router as material_matching_router
from api.endpoints.account.signUp import router as signUp_router
from api.endpoints.account.signIn import router as signIn_router
from api.endpoints.predictImport import router as predictImport_router

app = FastAPI()

@app.get("/")
async def index():
    return {"message": "Welcome to the API"}

@app.get("/docs")
async def docs():
    return {"message": "API documentation"}

@app.get("/openapi.json")

async def get_openapi_endpoint():
    return JSONResponse(get_openapi(title="Your API Title", version="1.0.0", routes=app.routes))

app.include_router(predict_router, prefix="/predict",tags=["Predictor"])
app.include_router(material_matching_router, prefix="/materialmatching",tags=["Material Matching"])
app.include_router(predictTest_router, prefix="/predictTest",tags=["Predict Testing"])
app.include_router(signUp_router, prefix="/signUp",tags=["Account"])
app.include_router(signIn_router, prefix="/signIn",tags=["Account"])
app.include_router(predictImport_router, prefix="/predictImport",tags=["Predict Testing"])

