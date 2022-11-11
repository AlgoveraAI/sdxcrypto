import os
import sys
import uuid
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, StableDiffusionImg2ImgPipeline

#CLIP STUFFS
sys.path.append('src/blip')
sys.path.append('src/clip')
sys.path.append('clip-interrogator')

import clip
import torch
from clip_interrogator import Interrogator, Config


#import from this lib
import firebase
from firebase import Bucket
from config import STORAGE_URL, GOOGLE_APPLICATION_CREDENTIALS
from utils import createLogHandler

logger = createLogHandler(__name__, 'logs.log')
TMPSTORAGE = os.environ.get('TMPSTORAGE')

class Inference:
    def __init__(self):
        #holds different pipe
        self.pipes = {}
        self.storage = Bucket()
        self.uid = None
        self.job_uuid = None
        self.hf_token = os.getenv('HF_TOKEN')
        self.base_model_path = "/home/ec2-user/huggingface"

        self.base_models = {
            'midjourney-v4': 'prompthero/midjourney-v4-diffusion',
            # 'stable-diffusion-v1-5':'runwayml/stable-diffusion-v1-5'
        }

        #start clip models
        logger.info("Setting up CLIP")
        self.ci = Interrogator(Config())

        #start sd models
        logger.info("Setting up SD model pipes")
        self.setup_sdmodels()

    def run_inference(self, params):
        #GET PROMPT PARAMETERS
        prompt = params['prompt']
        neg_prompt = params['neg_prompt']
        base_model = params['base_model']
        height = params['height']
        width = params['width']
        inf_steps = params['inf_steps']
        guidance_scale = params['guidance_scale']
        seed = params['seed']
        job_type = params['job_type']

        #GET OWNER & JOB_UUID
        self.uid = params['uid']
        self.job_uuid = params['job_uuid']

        if job_type in ['txt2img', 'img2img']:
            # model_loc = firebase.get_basemodels(model=base_model)['model_location']
            pipe = self.get_pipe(job_type, 'midjourney-v4', '')

        elif job_type in ['clip2img']:
            # model_loc = firebase.get_basemodels(model=base_model)['model_location']
            pipe = self.get_pipe('img2img', 'midjourney-v4', '')

        if job_type == 'txt2img':
            initimage = None

        else:
            initimage = Image.open(f"{TMPSTORAGE}/{self.job_uuid}.jpg")
            initimage = self.resize(width, height, initimage)

        if job_type in ['txt2img', 'img2img', 'clip2img']:
            if job_type == 'clip2img':
                clip_prompt = self.clip_interrogator(initimage)
                firebase.update_job(self.job_uuid, {'prompt':clip_prompt})
                prompt = clip_prompt

            #add additional prompt for midjourney https://huggingface.co/prompthero/midjourney-v4-diffusion
            if base_model == 'midjourney-v4': 
                prompt = prompt + ". mdjrny-v4 style"
            
            logger.info(f"Final prompt - {prompt}")

            image = self.inference(pipe=pipe, 
                                    prompt=prompt,
                                    neg_prompt=neg_prompt,
                                    height=height, 
                                    width=width, 
                                    inf_steps=inf_steps, 
                                    guidance_scale=guidance_scale, 
                                    seed=seed, 
                                    initimage=initimage,
                                )

            self.storage.put(image, self.uid, self.job_uuid)
            logger.info(f'Uploaded asset to storage')

        else:
            clip_prompt = self.clip_interrogator(initimage)
            logger.info(clip_prompt)
            firebase.set_clip_result(self.job_uuid, clip_prompt)
        
        #REMOVE INITIMAGE FROM TMPSTORAGE
        if job_type in ['img2img']:
            os.remove(f"{TMPSTORAGE}/{self.job_uuid}.jpg")

        #REDUCE CREDIT_ON_HOLD
        firebase.remove_on_hold_credit(self.uid)

    def get_pipe(self, job_type, base_model, bm_local_path):
        mn = f"{job_type}_{base_model}"

        if mn in list(self.pipes.keys()):
            logger.info(mn)
            return self.pipes[mn]

        else:
            if job_type == 'txt2img':
                pipe = self.txt2img(bm_local_path)
                self.pipes[mn] = pipe
                return pipe

            elif job_type in ['img2img', 'clip2img']:
                pipe = self.img2img(bm_local_path)
                self.pipes[mn] = pipe
                return pipe

    def txt2img(self, bm_local_path):

        # scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        pipe = StableDiffusionPipeline.from_pretrained(bm_local_path, local_files_only=True).to("cuda")

        #switch off nsfw checker
        pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        
        #enable_attention_slicing
        pipe.enable_attention_slicing()

        return pipe

    def img2img(self, bm_local_path):

        # scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(bm_local_path, local_files_only=True).to("cuda")

        #switch off nsfw checker
        pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
        
        #enable_attention_slicing
        pipe.enable_attention_slicing()

        return pipe

    def clip_interrogator(self, image):
        image = image.convert('RGB')
        return self.ci.interrogate(image)

    def inference(self, 
        pipe, 
        prompt, 
        neg_prompt,
        height=256, 
        width=256, 
        inf_steps=50, 
        guidance_scale=7.5, 
        seed=69, 
        initimage=None
    ):
        if not initimage:
            with torch.cuda.amp.autocast():
                images = pipe([prompt] * 1, 
                            negative_prompt=[neg_prompt]*1,
                            num_inference_steps=inf_steps, 
                            guidance_scale=guidance_scale,
                            height=height,
                            width=width,
                            seed=seed).images
            return images[0]

        else:
            with torch.cuda.amp.autocast():
                images = pipe([prompt] * 1, 
                            negative_prompt=[neg_prompt]*1,
                            num_inference_steps=inf_steps, 
                            guidance_scale=guidance_scale,
                            height=height,
                            width=width,
                            seed=seed,
                            init_image=initimage).images
            return images[0]     

    def resize(self, w_val, l_val, img):
        img = img.resize((w_val,l_val), Image.Resampling.LANCZOS)
        return img       

    def setup_sdmodels(self):
        
        for bm_name, bm_hf_path in self.base_models.items():
            bm_local_path = f"{self.base_model_path}/{bm_name}"

            for job_type in ['txt2img', 'img2img']:
                logger.info(f"Setting up pipe for base_model: {bm_name}, job_type: {job_type}")
                self.get_pipe(job_type, bm_name, bm_local_path)

        logger.info(f"SD Models in pipe {self.pipes.keys()}")



