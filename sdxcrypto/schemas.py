import json
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List

from pydantic.types import conint

class JobStatusOut(BaseModel):
    job_uuid: str
    job_status: str
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class BaseModelsOut(BaseModel):
    base_models: list[str] = []

class BaseModels(BaseModel):
    base_model_name: str
    model_location: str

class JobCreateTxt2ImgIn(BaseModel):
    prompt: str 
    uid: str
    base_model: str="stable-diffusion-v1-5"
    neg_prompt: str=""
    height: int = 512
    width: int = 512
    inf_steps:int = 50
    guidance_scale:float = 7.5
    seed: int = 69

class JobCreateImg2ImgIn(BaseModel):
    prompt: str 
    uid: str
    base_model: str="stable-diffusion-v1-5"
    neg_prompt: str=""
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

class JobCreateOut(BaseModel):
    job_uuid: str
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class JobCreateClipIn(BaseModel):
    uid: str
    prompt: str="NA"
    base_model: str="NA"
    neg_prompt: str="NA"
    height: int = 512
    width: int = 512
    inf_steps: str="NA"
    guidance_scale: str="NA"
    seed: str="NA"

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class JobCreateClip2ImgIn(BaseModel):
    uid: str
    base_model: str ="stable-diffusion-v1-5"
    prompt: str="FromClip"
    neg_prompt: str=""
    height: int=512
    width: int=512
    inf_steps: int=50
    guidance_scale: float = 7.5
    seed: int=69

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class AssetListIn(BaseModel):
    uid: str
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
        
class AssetListOut(BaseModel):
    assets: List[str]

class AssetGetIn(BaseModel):
    uid: str
    job_uuid: str
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class AddJob(BaseModel):
    job_uuid: str
    job_status: str
    job_created: str
    job_in_process: str
    job_done: str
    uid: str
    base_model: str
    prompt: str
    neg_prompt: str
    guidance_scale: float = 7.
    inf_steps: int = 50 
    height: int = 512
    width: int = 512
    seed: int = 36
