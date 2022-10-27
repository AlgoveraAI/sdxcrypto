import os
import uuid
import numpy as np
import uuid
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from sqlalchemy.orm import Session

#import from this lib
from config import settings
from models import models
from db.db import engine
from utils import createLogHandler

logger = createLogHandler(__name__, 'logs.log')

class Inference:
    def __init__(self, base_storage=None):
        #holds different pipe
        self.pipes = {}
        if not base_storage:
            self.base_storage = os.getcwd()
        else:
            self.base_storage = base_storage
        
        logger.info(f'Base storage set as - {self.base_storage}')
        
        self.owner_id = None
        self.job_uuid = None
        self.hf_token = settings.hf_token

    def run_inference(self, params):
        #get prompt parameters
        prompt = params.prompt
        neg_prompt = params.neg_prompt
        base_model = params.base_model
        num_samples = params.num_samples
        height = params.height
        width = params.width
        inf_steps = params.inf_steps
        guidance_scale = params.guidance_scale
        seed = params.seed

        #get owner_id & job_uuid
        self.owner_id = params.owner_id
        self.job_uuid = params.job_uuid

        #get pipe
        pipe = self.get_pipe(base_model)

        #run inference
        logger.info(f'Starting job with prompt: {prompt}')
        images = self.inference(pipe=pipe, 
                                prompt=prompt,
                                neg_prompt=neg_prompt,
                                num_samples=num_samples, 
                                height=height, 
                                width=width, 
                                inf_steps=inf_steps, 
                                guidance_scale=guidance_scale, 
                                seed=seed, 
                            )
        #create asset uuid
        asset_uuids = [uuid.uuid4().hex for i in range(num_samples)]

        #save to storage and update asset db
        self.save_to_storage(images, asset_uuids)

        return images

    def get_pipe(self, model_name):

        if not self.pipes:
            scheduler = LMSDiscreteScheduler(beta_start=0.00085, 
                                                beta_end=0.012, 
                                                beta_schedule="scaled_linear")
            
            pipe = StableDiffusionPipeline.from_pretrained(
                                                        model_name, 
                                                        scheduler=scheduler, 
                                                        use_auth_token=self.hf_token,
                                                        revision="fp16", 
                                                        torch_dtype=torch.float16).to("cuda")

            self.pipes['model_name'] = pipe
            return pipe
        else:
            return self.pipes['model_name']

    def inference(self, 
        pipe, 
        prompt, 
        neg_prompt,
        num_samples, 
        height=256, 
        width=256, 
        inf_steps=50, 
        guidance_scale=7.5, 
        seed=69, 
    ):
        all_images = [] 
        with torch.cuda.amp.autocast():
            images = pipe([prompt] * num_samples, 
                        negative_prompt=[neg_prompt]*num_samples,
                        num_inference_steps=inf_steps, 
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        seed=seed).images

            all_images.extend(images)
        
        return all_images
    
    def save_to_storage(self, images, asset_uuids):
        #make directory under owner_id
        path = Path(f'{self.base_storage}/{self.owner_id}/images')
        path.mkdir(parents=True, exist_ok=True)

        #save images update asset db
        for img, fn in zip(images, asset_uuids):
            filename = f'{path}/{fn}.jpg'
            img.save(filename)
            
            #prepare asset_info
            asset_info = {
                'owner_id':self.owner_id,
                'asset_uuid':fn,
                'job_uuid':self.job_uuid,
                'filename':filename
            }

            #add asset_info to db
            with Session(engine) as session:
                new_asset = models.Asset(**asset_info)
                session.add(new_asset)
                session.commit()
            logger.info(f'added asset - {asset_info["asset_uuid"]}')