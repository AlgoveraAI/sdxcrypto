
from fastapi import APIRouter, Depends, status, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from PIL import Image

#imports from this lib
from routers import oauth2
from db.db import get_db
from models import schemas, models
from utils import zipfiles, createLogHandler

logger = createLogHandler(__name__, 'logs.log')

router = APIRouter(
    prefix="/assets",
    tags=['Assets']
)

# @router.get("/", status_code=status.HTTP_200_OK, response_model=schemas.AssetsOut) #
# def get_assets(current_user: int = Depends(oauth2.get_current_user), 
#                db: Session = Depends(get_db)):
    
#     all_assets = db.query(models.Asset).filter(models.Asset.owner_id == current_user.id).all()
#     if all_assets:
#         all_images = [get_bytes_value(Image.open(each.filename)) for each in all_assets]
#         return schemas.AssetsOut(assets=all_images)

#     else:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
#                             detail=f"Currently, there are no generated images")

@router.get("/", status_code=status.HTTP_200_OK) #
def get_asset(job_id:schemas.AssetsIn, 
              current_user: int = Depends(oauth2.get_current_user), 
              db: Session = Depends(get_db)):

    #if all jobs
    job_id = job_id.job_id
    if job_id == 'all':
        jobids = db.query(models.Job).filter(models.Job.owner_id == current_user.id)
        jobids = [id_.job_uuid for id_ in jobids]

        all_assets = []
        for job in jobids:
            for asset in db.query(models.Asset).filter(models.Job.job_uuid == job):
                all_assets.append(asset.filename)
        
        return  zipfiles(all_assets)
    
    else:
        #check if a job_id's owner == current user
        job_params = db.query(models.Job).filter(models.Job.job_uuid == job_id).first()
        job_owner = job_params.owner_id

        if job_owner != current_user.id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail=f"Not authorized")

        if not db.query(models.Job).filter(models.Job.job_uuid == job_id).first():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Job ID {jobid} not found")
        
        all_assets = db.query(models.Asset).filter(models.Asset.job_uuid == job_id).all()
        logger.info(all_assets)
        filenames = [each.filename for each in all_assets]

        return  zipfiles(filenames)

@router.get("/asset/list", status_code=status.HTTP_200_OK) #
def get_asset(current_user: int = Depends(oauth2.get_current_user), 
              db: Session = Depends(get_db)):

    assets =  db.query(models.Asset).filter(models.Job.owner_id == current_user.id).all()

    if not assets:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No asset found")

    
    return {'assets': [asset.asset_uuid for asset in assets]}

@router.get("/asset/{asset_id}", status_code=status.HTTP_200_OK) #
def get_asset(asset_id:str,
              current_user: int = Depends(oauth2.get_current_user), 
              db: Session = Depends(get_db)):

    asset =  db.query(models.Asset).filter(models.Asset.asset_uuid == asset_id).first()
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Asset ID {asset_id} not found")
    
    asset_owner = asset.owner_id
    
    if asset_owner != current_user.id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail=f"Not authorized")

    asset = asset.filename

    return FileResponse(asset)