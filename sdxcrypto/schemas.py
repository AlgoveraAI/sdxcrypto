import json
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List

from pydantic.types import conint

class UserOut(BaseModel):
    kind: str
    idToken: str
    email: EmailStr
    refreshToken: str
    expiresIn: str
    localId: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOutSimple(BaseModel):
    id: str
    email: EmailStr

class GetId(BaseModel):
    uid: str

class JobStatusIn(BaseModel):
    job_uuid: str
    idToken:str
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

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

class JobCreateIn(BaseModel):
    base_model: str
    prompt: str 
    neg_prompt: str=""
    idToken: str
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
    idToken: str
    base_model: str=""
    prompt: str=""
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

class JobCreateClip2ImgIn(BaseModel):
    idToken: str
    base_model: str
    prompt: str=""
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
    idToken: str
    
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
    idToken: str
    job_uuid: str
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class AssetsGetIn(BaseModel):
    idToken: str
    
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
    owner_uuid: str
    base_model: str
    prompt: str
    neg_prompt: str
    guidance_scale: float = 7.
    inf_steps: int = 50 
    height: int = 512
    width: int = 512
    seed: int = 36
