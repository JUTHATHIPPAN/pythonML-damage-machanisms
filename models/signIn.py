from pydantic import BaseModel

class UserSignIn(BaseModel):
    username: str
    password: str