from fastapi import APIRouter
from models.test import test
from core.config import cursor

router = APIRouter()
material_id = 2
cursor.execute("SELECT * FROM [dbo].[tb_basic_material] WHERE id = %s", (material_id,))

rows = cursor.fetchall()
# print(rows)
@router.get('/')
def get_test():
    # Logic to retrieve users
    return rows
