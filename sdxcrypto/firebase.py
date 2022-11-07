import os
import io
import time
import json
import requests
from PIL import Image

import firebase_admin
from firebase_admin import db
from firebase_admin import auth
from firebase_admin.auth import InvalidIdTokenError

from fastapi import status, HTTPException, APIRouter

from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials

#imports from this library
from utils import createLogHandler
from schemas import BaseModels, AddJob
from config import API_KEY, DATABASE_URL, STORAGE_URL, GOOGLE_APPLICATION_CREDENTIALS

#START LOGGER
logger = createLogHandler(__name__, 'logs.log')

#START FIREBASE APP
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
default_fb_app = firebase_admin.initialize_app(
    options={
    'databaseURL': DATABASE_URL
    }
)

#UTILS
def decode_response(response):
    res_json = json.loads(response.content)
    logger.info(res_json)
    if response.status_code == 200:
        return res_json
    else:
        raise HTTPException(status_code=res_json['error']['code'], detail=res_json['error']['message'])

#SIGNUP, SIGNIN, GET_USER
# def get_user(uid):
#     logger.info('here')
#     return auth.get_user(uid.uid)

def signup(email, password):
    ept = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"

    json_data = {
        'email': email,
        'password': password,
        'returnSecureToken': True,
    }

    response = requests.post(ept, json=json_data)
    return decode_response(response)

def signin(email, password):
    ept = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    json_data = {
        'email': email,
        'password': password,
        'returnSecureToken': True,
    }
    response = requests.post(ept, json=json_data)
    return decode_response(response)

def verify_idToken(idToken):
    try:
        decoded_token = auth.verify_id_token(idToken)
        return decoded_token

    except InvalidIdTokenError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                            detail='Invalid tokenID')

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
                            detail=f'{e}')

def verify_owner(job_uuid, idOwner):
    requested_job = get_job(job_uuid)

    if requested_job['owner_uuid'] != idOwner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials for job_id: {job_uuid}")
    return True

def verify_idToken_ownerId(info):
    idToken = info['idToken']
    job_uuid = info['job_uuid']
    
    decoded_token = verify_idToken(idToken)
    ownerVerify = verify_owner(job_uuid, decoded_token['uid'])
    assert ownerVerify==True

    return decoded_token


def refresh_idToken(refreshToken):
    ept = f"https://securetoken.googleapis.com/v1/token?key={API_KEY}"
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = f'grant_type=refresh_token&refresh_token={refreshToken}'

    response = requests.post(ept, headers=headers, data=data)

    return json.loads(response.content)

#DATABASE
def set_basemodels(params:BaseModels):
    ref = db.reference('stablediffusion/base_models')
    ref.child(params['base_model_name']).set({'model_location':params['model_location']})
    return ref.get()

def get_basemodels(model=None):
    if model:
        ref = db.reference(f'stablediffusion/base_models/{model}')
        return ref.get()
    else:
        ref = db.reference('stablediffusion/base_models')
        return ref.get()

def update_basemodels(child, new_location):
    ref = db.reference(f'stablediffusion/base_models/{child}')
    ref.update({
        'model_location': new_location
    })
    return ref.get()

def delete_basemodels(child):
    ref = db.reference(f'stablediffusion/base_models/{child}')
    ref.delete()
    return ref.get()

def set_job(params:AddJob):
    ref = db.reference('stablediffusion/jobs')
    job_uuid = params.pop('job_uuid')
    ref.child(job_uuid).set({**params})
    return ref.get()

def get_job(job_uuid):
    ref = db.reference(f'stablediffusion/jobs/{job_uuid}')
    return ref.get()

def update_job(child, new_dict):
    ref = db.reference(f'stablediffusion/jobs/{child}')
    ref.update(
        new_dict
    )
    return ref.get()

def get_job_by_owner(owner_uuid):
    ref = db.reference('stablediffusion/jobs/')
    snapshot = ref.order_by_child('owner_uuid').equal_to(owner_uuid).get()
    jobs = [k for k,v in snapshot.items()]
    return jobs


#storage
class Bucket():
    def __init__(self):
        self.base_storage = STORAGE_URL
        scopes = [
                'https://www.googleapis.com/auth/firebase.database',
                'https://www.googleapis.com/auth/userinfo.email',
                "https://www.googleapis.com/auth/cloud-platform"
            ]
        self.credentials = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_APPLICATION_CREDENTIALS, scopes)
        self.url = f"https://firebasestorage.googleapis.com/v0/b/{STORAGE_URL}"
        self.client = storage.Client(credentials=self.credentials, project=STORAGE_URL)
        self.bucket = self.client.get_bucket(STORAGE_URL)

    def put(self, 
        image:Image, 
        owner_uuid:str, 
        job_uuid:str
    ):
        path = f'{owner_uuid}/images/{job_uuid}.jpg'
        blob = self.bucket.blob(path)
        blob.upload_from_string(self.get_bytes_value(image), content_type="image/jpeg")

    def get_bytes_value(self, 
        image:Image
    ):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

    def download(self,
        owner_uuid:str, 
        job_uuid:str
    ):
        blob = self.bucket.get_blob(f'{owner_uuid}/images/{job_uuid}.jpg')
        asset = blob.download_as_string()
        logger.info(asset)
        return asset