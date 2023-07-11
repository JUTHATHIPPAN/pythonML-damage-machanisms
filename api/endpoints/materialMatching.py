import joblib
from fastapi import APIRouter, HTTPException
from sklearn.feature_extraction.text import CountVectorizer
from core.config import cursor
from models.materialMatching import MaterailGradePredictionRequest, MaterailGradePredictionResponse

router = APIRouter()
clf = joblib.load('models_ml/material/MatModel.joblib')

# Initialize the CountVectorizer
vectorizer = CountVectorizer()
vectorizer.vocabulary_ = joblib.load('models_ml/material/VocabMatModel.joblib')
    
# Function to search material by grade
def search_material_by_grade(material_id):
    # Execute a SELECT query
    cursor.execute("SELECT * FROM [dbo].[tb_basic_material] WHERE id = %s", (material_id,))
    # Fetch all the rows returned by the query
    rows = cursor.fetchall()
    print(rows)
    return rows


@router.post("/")
def predict_damage(request: MaterailGradePredictionRequest):
    
    input_string = request.grade
    characters_to_remove = [" ", "/", "-", "."]
    for char in characters_to_remove:
        input_string = input_string.replace(char, "")

    # Transform the input string using the vectorizer
    input_vector = vectorizer.transform([input_string])
    prediction = clf.predict(input_vector)
    material_info = search_material_by_grade(str(prediction[0]))

    if material_info:
        response = MaterailGradePredictionResponse(
                                                    Grade_Material = material_info[0][1],
                                                    Base_Material = material_info[0][2],
                                                    Composition = material_info[0][3],
                                                    Base_Material_Abbreviation = material_info[0][4],
                                                    Ultimate_Tensile_Strength_ksi = material_info[0][5],
                                                    Material_ID = material_info[0][6],
                                                    )
        return response
    else:
        raise HTTPException(status_code=404, detail='No material found for the provided grade.')
