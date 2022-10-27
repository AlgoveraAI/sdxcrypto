import uuid
from typing import List
from sqlalchemy.orm import Session
from fastapi import FastAPI, Response, status, HTTPException, Depends, APIRouter

#imports from this lib
import utils
from db.db import get_db
from models import models, schemas
from routers import oauth2
import scheduler_interface 

router = APIRouter(
    prefix="/job",
    tags=['Generate']
)

@router.get("/basemodels", status_code=status.HTTP_200_OK, response_model=schemas.BaseModelsOut) #
def get_basemodels(db: Session = Depends(get_db)):
    # print([each.name for each in db.query(models.BaseModels).all()])
    return schemas.BaseModelsOut(base_models=[each.name for each in db.query(models.BaseModels).all()])

@router.post("/create", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
def create_job(job: schemas.JobCreateIn, 
               current_user: int = Depends(oauth2.get_current_user), 
               db: Session = Depends(get_db)):
    #create unique job uuid
    job_uuid = uuid.uuid4().hex
    
    #add to db - table jobs
    incoming_job = models.Jobs(job_uuid=job_uuid, 
                                owner_id=current_user.id, 
                                job_status='created', 
                                **job)
    db.add(incoming_job)
    db.commit()
    db.refresh(incoming_job)

    #add to queue
    scheduler_interface.schedule_job(job_uuid)

    return {"jobID": job_uuid}

@router.get("/status", status_code=status.HTTP_200_OK, response_model=schemas.JobStatusOut)
def status_job(jobID: schemas.JobStatusIn, 
               current_user: int = Depends(oauth2.get_current_user),
               db: Session = Depends(get_db)):

    #check if the job was created by the current user
    id_ = jobID['jobID']
    requested_job = db.query(models.Jobs).filter(models.Jobs.uuid == id_).first()

    if requested_job['owner_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials for job_id: {id_}")
    #retrieve job status
    else:
        status = requested_job['job_status']
    return {'jobID':id_, 'jobStatus': status}