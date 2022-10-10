import os
import zipfile
import glob
import shutil
import json
from pathlib import Path
from io import StringIO, BytesIO
from typing import Union, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, status
from fastapi.responses import StreamingResponse, Response 
from sdxcrypto.api.inference import Inference
from sdxcrypto.api.tracker import BaseModels
from sdxcrypto.api.training import Training

#requestbody for generate
class GenerateRequest(BaseModel):
    base_model: str
    prompt: str 
    num_samples: int = 1
    height: int = 512
    width: int = 512
    inf_steps:int = 50
    guidance_scale:float = 7.5
    seed: int = 69

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

#requestbody for train
class TrainRequest(BaseModel):
    base_model: str
    concept_name: str
    ins_prompt: str
    resolution: int
    prior: bool
    prior_prompt: str
    train_steps:int = 400

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

#responsemodel for train 
class TrainResponse(BaseModel):
    model_type: str
    model_name: str
    model_dir: str

#extend the app
app = FastAPI()

#extend infence and train modules
track = BaseModels()
inf = Inference()
trn =  Training()

@app.get("/")
def homepage():
    return "Algovera SD x Crypto API"

#endpoint for generating images
@app.post("/generate")
def generate(data: GenerateRequest = Form(...),
             files: Union[UploadFile, None] = None):
    
    params=dict(data)

    if files: params['img2img'] = True
    else: params['img2img'] = False
   
    cwd = os.getcwd()
    base_path = f"{cwd}/storage/init_images/"
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    if files:
        destination_file_path = base_path+files.filename #output file path
        with open(destination_file_path, 'wb') as out_file:
            shutil.copyfileobj(files.file, out_file)

    images = inf.run_inference(params)

    files = glob.glob(f"{base_path}/*")
    for f in files:
        os.remove(f)

    cwd = os.getcwd()
    zip_subdir  = f"{cwd}/storage/output_images"
    zip_io = BytesIO()

    with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as temp_zip:
        for fpath in glob.glob(f'{zip_subdir}/*'):
            # Calculate path for file in zip
            fdir, fname = os.path.split(fpath)
            zip_path = os.path.join(zip_subdir, fname)
            # Add file, at correct path
            temp_zip.write(fpath, zip_path)
    
    files = glob.glob(f"{zip_subdir}/*")

    for f in files:
        os.remove(f)

    return StreamingResponse(
        iter([zip_io.getvalue()]), 
        media_type="application/x-zip-compressed", 
        headers = { "Content-Disposition": f"attachment; filename=images.zip"})
    

#endpoint for textual inversion
@app.post("/train", response_model=TrainResponse)
def train(data: TrainRequest = Form(...), 
          files: List[UploadFile] = File(...)):
    
    params = dict(data)
    
    cwd = os.getcwd()
    base_path = f"{cwd}/storage/{params['concept_name']}/input_images/"
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    if files:
        for file in files:
            destination_file_path = base_path+file.filename #output file path
            with open(destination_file_path, 'wb') as out_file:
                shutil.copyfileobj(file.file, out_file)
    
    tosave = trn.run_training(params)
    return {'model_type':tosave[0], 'model_name':tosave[1], 'model_dir':tosave[2]}