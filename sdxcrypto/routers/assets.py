from fastapi import APIRouter, Depends, status, HTTPException, Response
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from PIL import Image

#imports from this lib
from routers import oauth2
from db.db import get_db
from models import schemas, models
from utils import get_bytes_value

router = APIRouter(
    prefix="/assets",
    tags=['Assets']
)

@router.get("/", status_code=status.HTTP_200_OK, response_model=schemas.AssetsOut) #
def get_assets(current_user: int = Depends(oauth2.get_current_user), 
               db: Session = Depends(get_db)):
    
    all_assets = db.query(models.Asset).filter(owner_id == current_user.id).all()
    if all_assets:
        all_images = [get_bytes_value(Image.open(each.filename)) for each in all_assets]
        return schemas.AssetsOut(assets=all_images)

    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Currently, there are no generated images")

@router.get("/{jobid}", status_code=status.HTTP_200_OK, response_model=schemas.AssetsOut) #
def get_asset(current_user: int = Depends(oauth2.get_current_user), 
              db: Session = Depends(get_db)):

    #check if a job_id's owner == current user
    job_params = db.query(models.Jobs).filter(job_uuid=jobid).first()
    job_owner = job_params['owner_id']

    if job_owner != current_user.id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=f"Not authorized")
    
    all_assets = db.query(models.Asset).filter(job_uuid == jobid).all()
    all_images = [get_bytes_value(Image.open(each.filename)) for each in all_assets]
    return  schemas.AssetsOut(assets=all_images)
