# from xmlrpc.client import Boolean
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql.expression import text
from sqlalchemy.sql.sqltypes import TIMESTAMP

#imports from this lib
from db.db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(
        Integer, 
        primary_key=True, 
        nullable=False
        )
    email = Column(
        String, 
        nullable=False, 
        unique=True
        )
    password = Column(
        String, 
        nullable=False
        )
    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False, 
        server_default=text('now()')
        )

class Jobs(Base):
    __tablename__ = "jobs"
    id = Column(
        Integer, 
        primary_key=True, 
        nullable=False)
    job_uuid = Column(
        String, 
        nullable=False
        )
    owner_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False
        )
    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False, 
        server_default=text('now()')
        )
    base_model = Column(
        String,
        nullable=False
    )
    prompt = Column(
        String,
    )
    neg_prompt = Column(
        String,
    )
    num_samples = Column(
        Integer
    )
    height = Column(
        Integer
    )
    width = Column(
        Integer
    )
    inf_steps = Column(
        Integer
    )
    guidance_scale = Column(
        Float
    )
    seed = Column(
        Integer
    )
    job_status = Column(
        String
    )

class Asset(Base):
    __tablename__ = "assets"

    id = Column(
        Integer, 
        primary_key=True, 
        nullable=False)
    
    owner_id = Column(
        String, 
        nullable=False
        )
    
    asset_uuid = Column(
        String, 
        nullable=False
        )

    job_uuid = Column(
        String, 
        nullable=False
        )

    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False, 
        server_default=text('now()')
        )

    filename = Column(
        String,
        nullable=False
    )

class BaseModels(Base):
    __tablename__ = "basemodels"
    id = Column(
        Integer, 
        primary_key=True, 
        nullable=False)
        
    name = Column(
        String,
        nullable=False
    )

# class Subscription(Base):
#     __tablename__ = "subscription"
#     pass