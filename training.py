import os
import argparse
from argparse import Namespace
import itertools
import math
from contextlib import nullcontext
import random
import numpy as np
import gc
import PIL
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from huggingface_hub.hf_api import HfApi

import bitsandbytes as bnb

from pathlib import Path

from utils import DreamBoothDataset, PromptDataset, training_function


#set the access token
hf_api = HfApi()
hf_api.USERNAME_PLACEHOLDER = "arshy"
hf_api.set_access_token(access_token=os.getenv("HF_ACCESS_TOKEN"))

#Arguments
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
instance_prompt = "beautiful sks hafsah" #os.getenv("INS_PROMPT")

prior_preservation = False 
prior_preservation_class_prompt = "beautiful girl, child" # os.getenv("PRIOR_PROMPT")
save_path = "input_images"

#Advanced settings for prior preservation (optional)
num_class_images = 12 
sample_batch_size = 2
prior_loss_weight = 1 
prior_preservation_class_folder = "./class_images" 
class_data_root=prior_preservation_class_folder

#generate class images
if(prior_preservation):
    class_images_dir = Path(class_data_root)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < num_class_images:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, use_auth_token=True, revision="fp16", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.enable_attention_slicing()
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = num_class_images - cur_class_images
        print(num_new_images)
        print(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

        for example in tqdm(sample_dataloader, desc="Generating class images"):
            with torch.autocast("cuda"):
                images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")
        pipeline = None
        gc.collect()
        del pipeline
        with torch.no_grad():
          torch.cuda.empty_cache()

# #Load the Stable Diffusion model
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="text_encoder", 
    use_auth_token=os.getenv("HF_ACCESS_TOKEN")
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="vae", 
    use_auth_token=os.getenv("HF_ACCESS_TOKEN")
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="unet", 
    use_auth_token=os.getenv("HF_ACCESS_TOKEN"),
    gradient_checkpointing=True
)

tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_auth_token=os.getenv("HF_ACCESS_TOKEN"),
)

args = Namespace(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    resolution=512,
    center_crop=True,
    instance_data_dir=save_path,
    instance_prompt=instance_prompt,
    learning_rate=5e-06,
    max_train_steps=400,
    train_batch_size=1,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    mixed_precision="no", # set to "fp16" for mixed-precision training.
    # gradient_checkpointing=True, # set this to True to lower the memory usage.
    use_8bit_adam=True, # use 8bit optimizer from bitsandbytes
    seed=3434554,
    with_prior_preservation=prior_preservation, 
    prior_loss_weight=prior_loss_weight,
    sample_batch_size=2,
    class_data_dir=prior_preservation_class_folder, 
    class_prompt=prior_preservation_class_prompt, 
    num_class_images=num_class_images, 
    output_dir="output",
)

accelerate.notebook_launcher(training_function, args=(args, text_encoder, vae, unet, tokenizer), num_processes=1)
with torch.no_grad():
    torch.cuda.empty_cache()

