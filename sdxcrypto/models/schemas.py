import json
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List

from pydantic.types import conint

class UserOut(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime

    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    id: Optional[str] = None

class JobCreateIn(BaseModel):
    base_model: str
    prompt: str 
    neg_prompt: str
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

class JobCreateOut(BaseModel):
    job_id: str
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class JobStatusIn(BaseModel):
    job_id: str
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class JobStatusOut(BaseModel):
    job_id: str
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

class AssetsIn(BaseModel):
    job_id: str

# class AssetsIDIn(BaseModel):
#     asset_id: str


class AssetsOut(BaseModel):
    assets: list[str] = []
