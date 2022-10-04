
import os 
import uuid
import pandas as pd
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

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

        pipe = self.get_pipe(base_model)
        
        images = self.inference(pipe, 
                                prompt, 
                                num_samples, 
                                height, 
                                width, 
                                inf_steps, 
                                guidance_scale, 
                                seed)
        return images

    def get_pipe(self, model_name):
        # check if there is already the pipe in pipes
        # if yes and the selected model is same return
        # if model different - set up and add to pipes and return pipe
        if model_name in self.pipes:
            return self.pipes[model_name]

        else:
            model_type = self.data[self.data["model_name"] == model_name]["model_type"].values[0]
            print(model_type)
            if model_type == "base_model":
                # scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
                pipe = StableDiffusionPipeline.from_pretrained(
                                                               model_name, 
                                                            #    scheduler=scheduler, 
                                                               use_auth_token=self.hf_token,
                                                               revision="fp16", 
                                                               torch_dtype=torch.float16).to("cuda")
                self.pipes[model_name] = pipe
                return pipe

            else:
                model_dir = self.data[self.data["model_name"] == model_name]["model_dir"].values[0]
                print(model_dir)
                pipe = StableDiffusionPipeline.from_pretrained(
                                                                model_dir,
                                                                revision="fp16", 
                                                                torch_dtype=torch.float16,
                                                            ).to("cuda")
                self.pipes[model_name] = pipe          
                return pipe

    def inference(self, pipe, prompt, num_samples, height=256, width=256, inf_steps=50, guidance_scale=7.5, seed=69):
        all_images = [] 
        with torch.autocast("cuda"):
            images = pipe([prompt] * num_samples, 
                          num_inference_steps=inf_steps, 
                          guidance_scale=guidance_scale,
                          height=height,
                          width=width,
                          seed=seed).images

            all_images.extend(images)

        self.mk_dir()

        [img.save(f"{self.image_output_dir }/{uuid.uuid4().hex}.jpg") for img in all_images]
        
        return all_images
    
    def mk_dir(self):
        if not os.path.exists(self.image_output_dir):
            os.mkdir(image_output_dir)
