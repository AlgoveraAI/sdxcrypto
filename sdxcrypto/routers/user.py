from fastapi import FastAPI, Response, status, HTTPException, APIRouter
from firebase_admin import auth

#imports from this lib
import utils
import schemas
from utils import createLogHandler
from firebase import (
                        signup, 
                        signin, 
                    )

router = APIRouter(
    prefix="/users",
    tags=['Users']
)

logger = createLogHandler(__name__, 'logs.log')

@router.post("/signup", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def signup_user(user: schemas.UserCreate):
    new_user = signup(user.email, user.password)
    return new_user

@router.post("/signin", status_code=status.HTTP_201_CREATED, response_model=schemas.UserOut)
def signin_user(user: schemas.UserLogin):
    new_user = signin(user.email, user.password)
    return new_user

@router.get('/get', response_model=schemas.UserOutSimple)
def get_user(GetId: schemas.GetId):
    try:
        user = auth.get_user(GetId.uid)
        return {'id': user.uid, 'email':user.email}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"{e}")
    