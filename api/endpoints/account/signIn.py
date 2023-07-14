from fastapi import APIRouter, HTTPException
from models.signIn import UserSignIn
from core.config import conn, JWT_SECRET_KEY, JWT_ALGORITHM
import bcrypt
import jwt

router = APIRouter()

@router.post('/')
def sign_in(user: UserSignIn):
    try:
        username = user.username
        password = user.password

        # Execute SQL query to fetch user details
        cursor = conn.cursor(as_dict=True)
        query = "SELECT username, password FROM [ML_damage].[dbo].[tb_account] WHERE username = %s"
        cursor.execute(query, (username,))
        result = cursor.fetchone()

        if result is None:
            raise HTTPException(status_code=401, detail='Invalid username or password')

        stored_password = result['password']

        # Verify password
        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            # Generate token
            token = generate_token(username)
            return {'message': 'Authentication successful', 'token': token}

        raise HTTPException(status_code=401, detail='Invalid username or password')

    except HTTPException:
        raise

    except Exception as e:
        print('Error:', str(e))
        raise HTTPException(status_code=500, detail='Internal Server Error')


def generate_token(username: str) -> str:
    payload = {'username': username}
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token
