
from fastapi import APIRouter, Depends, status, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from PIL import Image
import io

#imports from this lib
import schemas
import firebase
from utils import zipfiles, createLogHandler

logger = createLogHandler(__name__, 'logs.log')

router = APIRouter(
    prefix="/assets",
    tags=['Assets']
)

@router.get("/list", status_code=status.HTTP_200_OK, response_model=schemas.AssetListOut) 
def get_asset(reqbody:schemas.AssetListIn):
    #verify idToken
    decoded_token = firebase.verify_idToken(reqbody['idToken'])
    #get list of assets
    owner_uuid = decoded_token['uid']
    assets = firebase.get_job_by_owner(owner_uuid)

    return {'assets': assets}

@router.get("/getAsset", status_code=status.HTTP_200_OK) 
def get_asset(reqbody:schemas.AssetGetIn):
    #verify idToken and ownership
    decoded_token = firebase.verify_idToken_ownerId(reqbody)
    
    job_uuid = reqbody['job_uuid']
    job_params = firebase.get_job(job_uuid)
    job_type = job_params['job_type']

    if job_type in ['txt2img', 'img2img', 'clip2img']:
        #download asset from fb store
        buck = firebase.Bucket()
        asset = buck.download(decoded_token['uid'], reqbody['job_uuid'])
        def iterbyte():
            with io.BytesIO(asset) as f:
                yield from f
        # return FileResponse(io.BytesIO(asset))
        return StreamingResponse(iterbyte(), media_type="image/jpeg")

    else:
        prompt = firebase.get_clip_result(job_uuid)
        return prompt

@router.get("/getAssets", status_code=status.HTTP_200_OK) 
def get_assets(reqbody:schemas.AssetsGetIn):
    #verify idToken and ownership
    decoded_token = firebase.verify_idToken(reqbody['idToken'])

    #get assets from the owner
    assets = firebase.get_job_by_owner(decoded_token['uid'])

    if len(assets)==0:
        raise HTTPException(status_code=400, detail=f"No assets found for user with id {decoded_token['uid']}")
    
    elif len(assets)==1:
        #download asset from fb store
        buck = firebase.Bucket()
        asset = buck.download(decoded_token['uid'], assets[0])
        def iterbyte():
            with io.BytesIO(asset) as f:
                yield from f
        # return FileResponse(io.BytesIO(asset))
        return StreamingResponse(iterbyte(), media_type="image/jpeg")

    else:
        buck = firebase.Bucket()
        images = []
        for asset in assets:
            images.append((f'{asset}.jpg', buck.download(decoded_token['uid'], asset)))

        return zipfiles(images)
        
            

