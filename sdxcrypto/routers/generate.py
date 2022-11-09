import io
import os
import uuid
import json
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List
from fastapi import FastAPI, Response, status, HTTPException, APIRouter, File, UploadFile, Form

#imports from this lib
import utils
import schemas
import scheduler_interface 
import firebase
from utils import createLogHandler
router = APIRouter(
    prefix="/job",
    tags=['Generate']
)

logger = createLogHandler(__name__, 'logs.log')
TMPSTORAGE = os.environ.get('TMPSTORAGE')

@router.get("/basemodels", status_code=status.HTTP_200_OK, response_model=schemas.BaseModelsOut)
def get_basemodels():
    res = firebase.get_basemodels().keys()
    return {'base_models': list(res)}

@router.post("/create/txt2img", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
def create_job_txt2img(job: schemas.JobCreateIn):
    # verify idToken, (credit), add new job to db, add job_id to queue
    # verify idToken
    decoded_token = firebase.verify_idToken(job['idToken'])

    #create new job uuid
    job_uuid = uuid.uuid4().hex
    job['job_uuid'] = job_uuid
    job['job_status'] =  'created'
    job['job_created'] = str(datetime.timestamp(datetime.now()))
    job['job_in_process'] = ''
    job['job_done'] = ''
    job['owner_uuid'] = decoded_token['uid']
    job['job_type'] = 'txt2img'

    #add new job to job db
    firebase.set_job(job)  

    #add job_id to queue
    scheduler_interface.schedule_job(job_uuid)

    return {'job_uuid': job_uuid}

@router.post("/create/img2img", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
async def create_job_img2img(data: schemas.JobCreateIn=Form(...), 
                            image:UploadFile=File(...)):
    # verify idToken, (credit), add new job to db, add job_id to queue
    # verify idToken
    job = dict(data)
    decoded_token = firebase.verify_idToken(job['idToken'])

    #create new job uuid
    job_uuid = uuid.uuid4().hex
    job['job_uuid'] = job_uuid
    job['job_status'] =  'created'
    job['job_created'] = str(datetime.timestamp(datetime.now()))
    job['job_in_process'] = ''
    job['job_done'] = ''
    job['owner_uuid'] = decoded_token['uid']
    job['job_type'] = 'img2img'
    
    #add new job to job db
    firebase.set_job(job)    
    
    #receive and save image to tmpstorage
    img_obj = await image.read()
    img = Image.open(io.BytesIO(img_obj))
    img.save(f"{TMPSTORAGE}/{job_uuid}.jpg")

    #add job_id to queue
    scheduler_interface.schedule_job(job_uuid)

    return {'job_uuid': job_uuid}


@router.post("/create/clip", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
async def create_job_clip(data: schemas.JobCreateClipIn=Form(...), 
                            image:UploadFile=File(...)):
    # verify idToken, (credit), add new job to db, add job_id to queue
    # verify idToken
    job = dict(data)
    decoded_token = firebase.verify_idToken(job['idToken'])

    #create new job uuid
    job_uuid = uuid.uuid4().hex
    job['job_uuid'] = job_uuid
    job['job_status'] =  'created'
    job['job_created'] = str(datetime.timestamp(datetime.now()))
    job['job_in_process'] = ''
    job['job_done'] = ''
    job['owner_uuid'] = decoded_token['uid']
    job['job_type'] = 'clip-interrogator'
    
    #add new job to job db
    firebase.set_job(job)    
    
    #receive and save image to tmpstorage
    img_obj = await image.read()
    img = Image.open(io.BytesIO(img_obj))
    img.save(f"{TMPSTORAGE}/{job_uuid}.jpg")

    #add job_id to queue
    scheduler_interface.schedule_job(job_uuid)

    return {'job_uuid': job_uuid}

@router.post("/create/clip2img", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
async def create_job_clip2img(data: schemas.JobCreateClip2ImgIn=Form(...), 
                          image:UploadFile=File(...)):
    # verify idToken, (credit), add new job to db, add job_id to queue
    # verify idToken
    job = dict(data)
    decoded_token = firebase.verify_idToken(job['idToken'])

    #create new job uuid
    job_uuid = uuid.uuid4().hex
    job['job_uuid'] = job_uuid
    job['job_status'] =  'created'
    job['job_created'] = str(datetime.timestamp(datetime.now()))
    job['job_in_process'] = ''
    job['job_done'] = ''
    job['owner_uuid'] = decoded_token['uid']
    job['job_type'] = 'clip2img'
    
    #add new job to job db
    firebase.set_job(job)    
    
    #receive and save image to tmpstorage
    img_obj = await image.read()
    img = Image.open(io.BytesIO(img_obj))
    img.save(f"{TMPSTORAGE}/{job_uuid}.jpg")

    #add job_id to queue
    scheduler_interface.schedule_job(job_uuid)

    return {'job_uuid': job_uuid}

@router.get("/status", status_code=status.HTTP_200_OK, response_model=schemas.JobStatusOut)
def status_job(reqbody: schemas.JobStatusIn):
    # verify idToken
    #check if the job was created by the current user
    decoded_token = firebase.verify_idToken_ownerId(reqbody)
    
    #retrieve job status
    uuid = reqbody['job_uuid']
    requested_job = firebase.get_job(uuid)
    job_status = requested_job['job_status']

    return {'job_uuid':uuid, 'job_status': job_status}
