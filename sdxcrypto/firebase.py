import os
import io
import time
import json
import requests
from PIL import Image

import firebase_admin
from firebase_admin import db
from firebase_admin import firestore

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
default_fb_app = firebase_admin.initialize_app()
db = firestore.client()

db_basemodels = db.collection('basemodels')
db_jobs = db.collection('jobs')
db_users = db.collection('users')
db_clip = db.collection('clipprompt')

#UTILS
# def decode_response(response):
#     res_json = json.loads(response.content)
#     logger.info(res_json)
#     if response.status_code == 200:
#         return res_json
#     else:
#         raise HTTPException(status_code=res_json['error']['code'], detail=res_json['error']['message'])

#SIGNUP, SIGNIN, GET_USER
# def get_user(uid):
#     logger.info('here')
#     return auth.get_user(uid.uid)

# def signup(email, password):
#     ept = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"

#     json_data = {
#         'email': email,
#         'password': password,
#         'returnSecureToken': True,
#     }

#     response = requests.post(ept, json=json_data)
#     return decode_response(response)

# def signin(email, password):
#     ept = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
#     json_data = {
#         'email': email,
#         'password': password,
#         'returnSecureToken': True,
#     }
#     response = requests.post(ept, json=json_data)
#     return decode_response(response)

# def verify_idToken(idToken):
#     try:
#         decoded_token = auth.verify_id_token(idToken)
#         return decoded_token

#     except InvalidIdTokenError:
#         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
#                             detail='Invalid tokenID')

#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, 
#                             detail=f'{e}')

def verify_owner(job_uuid, idOwner):
    requested_job = get_job(job_uuid)

    if requested_job['owner_uuid'] != idOwner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid Credentials for job_id: {job_uuid}")
    return True

# def verify_idToken_ownerId(info):
#     idToken = info['idToken']
#     job_uuid = info['job_uuid']
    
#     decoded_token = verify_idToken(idToken)
#     ownerVerify = verify_owner(job_uuid, decoded_token['uid'])
#     assert ownerVerify==True

#     return decoded_token


# def refresh_idToken(refreshToken):
#     ept = f"https://securetoken.googleapis.com/v1/token?key={API_KEY}"
#     headers = {
#     'Content-Type': 'application/x-www-form-urlencoded',
#     }
#     data = f'grant_type=refresh_token&refresh_token={refreshToken}'

#     response = requests.post(ept, headers=headers, data=data)

#     return json.loads(response.content)

#DATABASE
def set_basemodels(params, db=db_basemodels):
    ref = db.document(f'{params["model_name"]}')
    ref.set({'model_name':params['model_name'],
             'model_location':params['model_location']})
    return ref.get().to_dict()

def get_basemodels(model=None, db=db_basemodels):
    if not model:
        return [{e.id: e.to_dict()} for e in db.get()]
    else:
        ref = db.document(f'{model}')
        return ref.get().to_dict()

def update_basemodels(params, db=db_basemodels):
    ref = db.document(f'{params["model_name"]}')
    ref.set({'model_name':params['model_name'],
             'model_location':params['model_location']})
    return ref.get().to_dict()

def delete_basemodels(model_name, db=db_basemodels):
    ref = db.document(model_name)
    ref.delete()
    return ref.get().to_dict()

def set_job(params, db=db_jobs):
    job_uuid = params.pop('job_uuid')
    ref = db.document(job_uuid)
    ref.set(params)
    return ref.get().to_dict()

def get_job(job_uuid, db=db_jobs):
    ref = db.document(job_uuid)
    return ref.get().to_dict()

def update_job(job_uuid, update_dict, db=db_jobs):
    ref = db.document(job_uuid)
    ref.update(update_dict)
    return ref.get().to_dict()

def get_job_by_owner(owner_uuid, db=db_jobs):
    return [{e.id: e.to_dict()} for e in db.where('owner_uuid', '==', owner_uuid).stream()]

def set_clip_result(job_uuid, prompt, db=db_clip):    
    ref = db.document(job_uuid)
    ref.set({'prompt':prompt})
    return ref.get().to_dict()

def get_clip_result(job_uuid, db=db_clip):
    ref = db.document(job_uuid)
    return ref.get().to_dict()

def get_credits(user):
    try:
        credits = user['credits']
    except:
        credits = 0
        
    try:
        credits_on_hold = user['credits_on_hold']
    except:
        credits_on_hold = 0
        
    return credits, credits_on_hold

def verify_hold_credit(uid, db=db_users):
    ref = db.document(uid)
    user = ref.get().to_dict()
    
    #CHECK USER EXISTS
    users = [e.id for e in db_users.list_documents()]
    if not uid in users:
        return status.HTTP_403_FORBIDDEN, "Unknown user"

    #GET CREDITS AND CREDITS_ON_HOLD
    credits, credits_on_hold = get_credits(user)
    
    #RAISE IF CREDITS LESS THAN 0
    if credits <= 0:
        return status.HTTP_403_FORBIDDEN, "Insufficient credits"
                            
    #MOVE A CREDIT FROM CREDIT TO ON_HOLD_CREDIT
    try:
        new_credits = credits - 1
        new_credits_on_hold = credits_on_hold + 1
        ref.update({'credits': new_credits, 'credits_on_hold':new_credits_on_hold})
    
        return 200, None
    except:
        return HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error"          

def reverse_hold_credit(uid, db=db_users):
    try:
        ref = db.document(uid)
        user = ref.get().to_dict()

        #GET CREDITS AND CREDITS_ON_HOLD
        credits, credits_on_hold = get_credits(user)
        
        #REVERSE CREDIT
        new_credits = credits + 1
        new_credits_on_hold = credits_on_hold - 1
        ref.update({'credits': new_credits, 'credits_on_hold':new_credits_on_hold})

    except:
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error"    


def remove_on_hold_credit(uid, db=db_users):
    try:
        ref = db.document(uid)
        user = ref.get().to_dict()
        new_credits_on_hold = user['credits_on_hold'] - 1
        
        ref.update({'credits_on_hold':new_credits_on_hold})
        
        return ref.get().to_dict()
    except:
        return status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error"

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
        path = f'{owner_uuid}/images/{job_uuid}/{job_uuid}.jpg'
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
        blob = self.bucket.get_blob(f'{owner_uuid}/images/{job_uuid}/{job_uuid}.jpg')
        asset = blob.download_as_string()
        logger.info(asset)
        return asset