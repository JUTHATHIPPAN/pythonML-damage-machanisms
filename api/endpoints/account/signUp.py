from fastapi import APIRouter, HTTPException
from models.signUp import UserCreate
from core.config import conn
import bcrypt

router = APIRouter()

# Hash password (register)
salt_rounds = 10

@router.post('/')
def sign_up(user: UserCreate):
    try:
        name = user.name
        username = user.username
        password = user.password
        is_active = 1

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(salt_rounds))

        # Execute SQL query
        cursor = conn.cursor(as_dict=True)
        query = "INSERT INTO [ML_damage].[dbo].[tb_account] (name, username, password, IsActive) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (name, username, hashed_password.decode('utf-8'), is_active))
        conn.commit()

        return {'message': 'User registered successfully'}

    except Exception as e:
        print('Error:', str(e))
        raise HTTPException(status_code=500, detail='Internal Server Error')
