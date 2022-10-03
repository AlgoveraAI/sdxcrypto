
import os 
import uuid
from PIL import Image
import pandas as pd

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

class Inference:
    def __init__(self):
        #holds different pipe
        self.pipes = {}
        self.data = pd.read_csv("storage/data.csv")
        self.hf_token = "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"

    def run_inference(self):
        prompt, model_name, num_samples = self.get_params()
        pipe = self.get_pipe(model_name)
        images = self.inference()
    
    def get_params(self):
        prompt = os.environ["PROMPT"]
        option = os.environ["OPTION"] 
        num_samples = int(os.environ["NUM_SAMPLES"])
        return prompt, option, num_samples

    def get_pipe(self, model_name):
        # check if there is already the pipe in pipes
        # if yes and the selected model is same return
        # if model different - set up and add to pipes and return pipe
        if model_name in self.pipes.keys():
            return self.pipes[model_name]

        else:
            model_type = self.data[self.data["model_name"] == model_name]["model_type"][0]
            print(model_type)
            if model_type == "base_model":
                scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
                pipe = StableDiffusionPipeline.from_pretrained(model_name, 
                                                               scheduler=scheduler, 
                                                               use_auth_token=self.hf_token)
                self.pipes[model_name] = pipe
                return pipe
            else:
                model_dir = self.data[self.data["model_name"] == model_name]["model_dir"]

                pipe = StableDiffusionPipeline.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,
                ).to("cuda")
                self.pipes[model_name] = pipe
                return pipe

    def inference(self, pipe, prompt, num_samples):
        all_images = [] 
        with torch.autocast("cuda"):
                images = pipe([prompt] * num_samples, num_inference_steps=50, guidance_scale=7.5).images
                all_images.extend(images)
        self.mk_dir()
        [img.save(f"{uuid.uuid4().hex}.jpg") for img in all_images]
        return all_images
    
    def mk_dir(self):
        image_output_dir = "storage/output_images"
        if not os.path.exists(image_output_dir):
            os.mk_dir(image_output_dir)
