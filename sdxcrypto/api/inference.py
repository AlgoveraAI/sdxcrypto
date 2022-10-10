import os
import io
import glob
import numpy as np
import uuid
import pandas as pd
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler

class Inference:
    def __init__(self):
        #holds different pipe
        self.cwd =os.getcwd()
        self.pipes = {}
        self.data = pd.read_csv(f"{self.cwd}/storage/data.csv")
        self.image_output_dir = f"{self.cwd}/storage/output_images"
        self.hf_token = "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"

    def run_inference(self, params):
        prompt = params["prompt"]
        base_model = params["base_model"]
        num_samples = params["num_samples"]
        height = params["height"]
        width = params["width"]
        inf_steps = params["inf_steps"]
        guidance_scale = params["guidance_scale"]
        seed = params["seed"]
        img2img = params["img2img"]

        pipe = self.get_pipe(base_model, img2img=img2img)
        
        if img2img:
            init_image = Image.open(glob.glob(f"{self.cwd}/storage/init_images/*")[0])
            print(init_image)
        
        else:
            init_image = None

        images = self.inference(pipe, 
                                prompt, 
                                num_samples, 
                                height, 
                                width, 
                                inf_steps, 
                                guidance_scale, 
                                seed, 
                                img2img, 
                                init_image)
        return images

    def get_pipe(self, model_name, img2img):
        # check if there is already the pipe in pipes
        # if yes and the selected model is same return
        # if model different - set up and add to pipes and return pipe
        if f"{model_name}_{img2img}" in self.pipes:
            print("model_name in pipe")
            print(f"{model_name}_{img2img}")

            return self.pipes[f"{model_name}_{img2img}"]

        else:
            model_type = self.data[self.data["model_name"] == model_name]["model_type"].values[0]
            if model_type == "base_model":
                if not img2img:
                    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
                    pipe = StableDiffusionPipeline.from_pretrained(
                                                                model_name, 
                                                                scheduler=scheduler, 
                                                                use_auth_token=self.hf_token,
                                                                revision="fp16", 
                                                                torch_dtype=torch.float16).to("cuda")

                else:
                    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                                                                model_name, 
                                                                scheduler=scheduler, 
                                                                use_auth_token=self.hf_token,
                                                                revision="fp16", 
                                                                torch_dtype=torch.float16).to("cuda")

            else:
                model_dir = self.data[self.data["model_name"] == model_name]["model_dir"].values[0]

                if not img2img:
                    pipe = StableDiffusionPipeline.from_pretrained(
                                                                    model_dir,
                                                                    revision="fp16", 
                                                                    torch_dtype=torch.float16,
                                                                ).to("cuda")
                else:
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                                                                    model_dir,
                                                                    revision="fp16", 
                                                                    torch_dtype=torch.float16,
                                                                ).to("cuda")
            
            self.pipes[f"{model_name}_{img2img}"] = pipe          
            return pipe

    def inference(self, pipe, prompt, num_samples, height=256, width=256, inf_steps=50, guidance_scale=7.5, seed=69, img2img=False, init_image=None):
        all_images = [] 
        if not img2img:
            print("not img2img")
            with torch.cuda.amp.autocast():
                images = pipe([prompt] * num_samples, 
                            num_inference_steps=inf_steps, 
                            guidance_scale=guidance_scale,
                            height=height,
                            width=width,
                            seed=seed).images

                all_images.extend(images)
        
        else:
            print("img2img")
            with torch.cuda.amp.autocast():
                images = pipe([prompt] * num_samples, 
                            init_image=init_image,
                            strength=0.6,
                            num_inference_steps=inf_steps, 
                            guidance_scale=guidance_scale,
                            seed=seed).images

                all_images.extend(images)
        self.mk_dir()
        [img.save(f"{self.image_output_dir }/{uuid.uuid4().hex}.jpg") for img in all_images]
        
        return all_images
    
    def mk_dir(self):
        if not os.path.exists(self.image_output_dir):
            os.mkdir(self.image_output_dir)

    def image_to_np(self, image: Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        return byte_im
