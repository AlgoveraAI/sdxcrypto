
import os 
import uuid
from PIL import Image
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

class Inference:
    def __init__(self):
        #holds different pipe
        self.pipes = {}
        self.data = pd.read_csv("storage/data.csv")
        self.hf_token = "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"
        self.image_output_dir = "storage/output_images"

    def run_inference(self):
        prompt, model_name, num_samples, height, width, inf_steps, guidance_scale, seed = self.get_params()
        pipe = self.get_pipe(model_name)
        images = self.inference(pipe, prompt, num_samples, height, width, inf_steps, guidance_scale, seed)
    
    def get_params(self):
        prompt = os.getenv("PROMPT")
        option = os.getenv("OPTION") 
        num_samples = int(os.getenv("NUM_SAMPLES"))
        height = int(os.getenv("HEIGHT"))
        width = int(os.getenv("WIDTH"))
        inf_steps = int(os.getenv("INF_STEPS"))
        guidance_scale = int(os.getenv("GUIDANCE_SCALE"))
        seed = int(os.getenv("SEED"))
        print(prompt, option)
        return prompt, option, num_samples, height, width, inf_steps, guidance_scale, seed

    def get_pipe(self, model_name):
        # check if there is already the pipe in pipes
        # if yes and the selected model is same return
        # if model different - set up and add to pipes and return pipe
        print(self.pipes)
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
            print(prompt)
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
