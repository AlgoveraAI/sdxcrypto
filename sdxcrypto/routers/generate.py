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
IMAGEJOBS = ['img2img', 'clip-interrogator', 'clip2img']

def execute_job(job, job_type, img=None):
    #CREATE A NEW JOB_UUID 
    job_uuid = uuid.uuid4().hex

    if job_type in IMAGEJOBS:
        #CONVERT PYDANTIC MODEL TO DICT
        job = json.loads(job.json())
        
        #RECEIVE AND SAVE INPUT IMAGE TO TMPSTORAGE
        img.save(f"{TMPSTORAGE}/{job_uuid}.jpg")

    else:
        #CONVERT PYDANTIC MODEL TO DICT
        job = dict(job)
        
    status_code, msg = firebase.verify_hold_credit(job['uid'])
    
    if not status_code == 200:
        if job_type in IMAGEJOBS: os.remove(f"{TMPSTORAGE}/{job_uuid}.jpg")
        raise HTTPException(
                status_code=status_code, detail=msg)
    
    #OTHER REQUIRED PARAMS
    job['job_uuid'] = job_uuid
    job['job_status'] =  'created'
    job['job_created'] = str(datetime.timestamp(datetime.now()))
    job['job_in_process'] = ''
    job['job_done'] = ''
    job['job_type'] = job_type
    
    #ADD NEW JOB TO JOB_DB
    try:
        firebase.set_job(job)

    except:
        if job_type in IMAGEJOBS: os.remove(f"{TMPSTORAGE}/{job_uuid}.jpg")
        firebase.reverse_hold_credit(job_uuid )

        raise HTTPException(
              status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

    #ADD JOB_UUID TO QUEUE
    try:
        scheduler_interface.schedule_job(job_uuid)

    except:
        if job_type in IMAGEJOBS: os.remove(f"{TMPSTORAGE}/{job_uuid}.jpg")
        firebase.reverse_hold_credit(job_uuid)

        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

    return job_uuid        


@router.get("/basemodels", status_code=status.HTTP_200_OK, response_model=schemas.BaseModelsOut)
def get_basemodels():
    try:
        res = firebase.get_basemodels().keys()
        return {'base_models': list(res)}
    except:
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")        

@router.post("/create/txt2img", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
def create_job_txt2img(job: schemas.JobCreateTxt2ImgIn):
    job_uuid = execute_job(job, 'txt2img')
    return {'job_uuid': job_uuid}

@router.post("/create/img2img", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
async def create_job_img2img(data: schemas.JobCreateImg2ImgIn=Form(...), 
                            image:UploadFile=File(...)):
    try:
        img_obj = await image.read()
        img = Image.open(io.BytesIO(img_obj))
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error")

    job_uuid = execute_job(data, 'img2img', img)
    return {'job_uuid': job_uuid}

@router.post("/create/clip", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
async def create_job_clip(data: schemas.JobCreateClipIn=Form(...), 
                            image:UploadFile=File(...)):
    try:
        img_obj = await image.read()
        img = Image.open(io.BytesIO(img_obj))
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error")

    job_uuid = execute_job(data, 'clip-interrogator', img)
    return {'job_uuid': job_uuid}

@router.post("/create/clip2img", status_code=status.HTTP_201_CREATED, response_model=schemas.JobCreateOut)
async def create_job_clip2img(data: schemas.JobCreateClip2ImgIn=Form(...), 
                             image:UploadFile=File(...)):
    try:
        img_obj = await image.read()
        img = Image.open(io.BytesIO(img_obj))
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error")

    job_uuid = execute_job(data, 'clip2img', img)
    return {'job_uuid': job_uuid}

@router.get("/status", status_code=status.HTTP_200_OK, response_model=schemas.JobStatusOut)
def status_job(job_uuid:str):
    #retrieve job status
    try:
        requested_job = firebase.get_job(job_uuid)
        job_status = requested_job['job_status']
        return {'job_uuid':job_uuid, 'job_status': job_status}
    except:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error")        

