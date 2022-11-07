import os
import uuid
import numpy as np
import uuid
from pathlib import Path
from PIL import Image
import torch

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

#import from this lib
import firebase
from firebase import Bucket
from config import STORAGE_URL, GOOGLE_APPLICATION_CREDENTIALS
from utils import createLogHandler

logger = createLogHandler(__name__, 'logs.log')

class Inference:
    def __init__(self):
        #holds different pipe
        self.pipes = {}
        self.storage = Bucket()
        self.owner_uuid = None
        self.job_uuid = None
        self.hf_token = os.getenv('HF_TOKEN')

    def run_inference(self, params):
        #get prompt parameters
        prompt = params['prompt']
        neg_prompt = params['neg_prompt']
        base_model = params['base_model']
        height = params['height']
        width = params['width']
        inf_steps = params['inf_steps']
        guidance_scale = params['guidance_scale']
        seed = params['seed']

        #get owner_id & job_uuid
        self.owner_id = params['owner_uuid']
        self.job_uuid = params['job_uuid']

        #get pipe
        model_loc = firebase.get_basemodels(model=base_model)['model_location']
        pipe = self.get_pipe(model_loc)

        #run inference
        logger.info(f'Starting job with prompt: {prompt}')
        image = self.inference(pipe=pipe, 
                                prompt=prompt,
                                neg_prompt=neg_prompt,
                                height=height, 
                                width=width, 
                                inf_steps=inf_steps, 
                                guidance_scale=guidance_scale, 
                                seed=seed, 
                            )

        #save to storage and update asset db
        self.storage.put(image, self.owner_id, self.job_uuid)
        logger.info(f'Uploaded asset to storage')
        return image

    def get_pipe(self, model_name):
        if model_name in list(self.pipes.keys()):
            logger.info(model_name)
            return self.pipes[model_name]

        else:
            scheduler = LMSDiscreteScheduler(beta_start=0.00085, 
                                             beta_end=0.012, 
                                             beta_schedule="scaled_linear")
            
            pipe = StableDiffusionPipeline.from_pretrained(model_name, 
                                                           scheduler=scheduler, 
                                                           use_auth_token=self.hf_token,
                                                           revision="fp16", 
                                                           torch_dtype=torch.float16).to("cuda")
            
            #switch off nsfw checker
            pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            #enable_attention_slicing
            pipe.enable_attention_slicing()
            self.pipes[model_name] = pipe
            
            return pipe

    def inference(self, 
        pipe, 
        prompt, 
        neg_prompt,
        height=256, 
        width=256, 
        inf_steps=50, 
        guidance_scale=7.5, 
        seed=69, 
    ):
        with torch.cuda.amp.autocast():
            images = pipe([prompt] * 1, 
                        negative_prompt=[neg_prompt]*1,
                        num_inference_steps=inf_steps, 
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        seed=seed).images
        return images[0]