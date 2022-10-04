from fastapi import FastAPI, File, UploadFile
from typing import Union, List
from pydantic import BaseModel

from sdxcrypto.api.inference import Inference
from sdxcrypto.api.tracker import BaseModels
from sdxcrypto.api.training import Training

#requestbody for generate
class GenerateBody(BaseModel):
    base_model: str
    prompt: str 
    num_samples: int = 1
    height: int = 512
    width: int = 512
    inf_steps:int = 50
    guidance_scale:float = 7.5
    seed: int = 69

#extend the app
app = FastAPI()

#extend infence and train modules
track = BaseModels()
inf = Inference()
trn =  Training()

@app.get("/")
def homepage():
    return "Algovera SD x Crypto API!"

#endpoint for generating images
@app.post("/generate")
def generate(params:GenerateBody):
    images = inf.run_inference(params=params)
    return images