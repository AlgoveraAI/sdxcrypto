import os
import sys
import pathlib
import subprocess
from pathlib import Path
import pkg_resources

from utils import createLogHandler

import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, StableDiffusionImg2ImgPipeline


logger = createLogHandler(__name__, 'logs.log') 

def initial_setup():
    #install requirements 
    logger.info("Installing requirements")
    with pathlib.Path('../requirements.txt').open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]

    #install CLIP stuffs
    if not Path('src').exists():
        logger.info("Preparing CLIP")
        install_cmds = [
            # ['pip', 'install', 'ftfy', 'regex', 'tqdm', 'timm', 'fairscale', 'requests'],
            ['pip', 'install', '-e', 'git+https://github.com/moarshy/CLIP.git@main#egg=clip'],
            ['pip', 'install', '-e', 'git+https://github.com/moarshy/BLIP.git@lib#egg=blip'],
            ['git', 'clone', 'https://github.com/moarshy/clip-interrogator.git']
        ]
        for cmd in install_cmds:
            print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))
    else:
        logger.info("CLIP preparation exists. Please check")

    sys.path.append('src/blip')
    sys.path.append('src/clip')
    sys.path.append('clip-interrogator')

class HFModels():
    def __init__(self):
        self.base_model_path = "/home/ec2-user/huggingface"
        self.HF_TOKEN = os.getenv('HF_TOKEN')
        self.base_models = {
            'midjourney-v4': 'prompthero/midjourney-v4-diffusion',
            'stable-diffusion-v1-5':'runwayml/stable-diffusion-v1-5'
        }

    def init_sdmodels(self):
        for bm_name, bm_hf_path in self.base_models.items():
            bm_local_path = f"{self.base_model_path}/{bm_name}"
            if not Path(bm_local_path).exists():
                if bm_name == "midjourney-v4": revision='main'
                else: revision="fp16"
                scheduler = LMSDiscreteScheduler(beta_start=0.00085, 
                                                    beta_end=0.012, 
                                                    beta_schedule="scaled_linear")
                
                pipe = StableDiffusionPipeline.from_pretrained(bm_hf_path, 
                                                            scheduler=scheduler, 
                                                            use_auth_token=self.HF_TOKEN,
                                                            revision=revision, 
                                                            torch_dtype=torch.float16).to("cuda")

                pipe.save_pretrained(bm_local_path)
                print('saved model')

    def test(self):
        for bm_name, bm_hf_path in self.base_models.items():
            bm_local_path = f"{self.base_model_path}/{bm_name}"
            pipe = StableDiffusionPipeline.from_pretrained(bm_local_path, local_files_only=True).to("cuda")

            prompt = "cat"
            neg_prompt = ""
            height=512
            width=512 
            inf_steps=10 
            guidance_scale=7.5 
            seed=69
            with torch.cuda.amp.autocast():
                images = pipe([prompt] * 1,
                                negative_prompt=[neg_prompt]*1,
                                num_inference_steps=inf_steps, 
                                guidance_scale=guidance_scale,
                                height=height,
                                width=width,
                                seed=seed).images

if __name__ == '__main__':
    import uvicorn 
    
    initial_setup()
    
    hfmodel = HFModels()
    hfmodel.init_sdmodels()
    
    uvicorn.run("main:app", host="0.0.0.0", port=8502, reload=False)