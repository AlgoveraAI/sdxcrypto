import os
import argparse
from argparse import Namespace
import itertools
import math
from contextlib import nullcontext
import random
import numpy as np
import pandas as pd
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
from huggingface_hub import hf_api

import bitsandbytes as bnb

from pathlib import Path

from .utils import DreamBoothDataset, PromptDataset, training_function
from .tracker import BaseModels

class Training:
    def __init__(self):
        self.access_token="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"
        self.concept_name = None
        self.cwd = os.getcwd() 
        self.tracker = BaseModels()

    def set_params(self, params):
        
        #Arguments
        pretrained_model_name_or_path = params['base_model']
        self.concept_name = params['concept_name']
        ins_prompt = params['ins_prompt']
        resolution = params['resolution']
        prior_preservation = params['prior'] 
        prior_preservation_class_prompt = params['prior_prompt']
        instance_data_dir = f"{self.cwd}/storage/{self.concept_name}/input_images"
        output_dir = f"{self.cwd}/storage/{self.concept_name}/output" 

        #Advanced settings for prior preservation (optional)
        num_class_images = 12 
        sample_batch_size = 2
        prior_loss_weight = 1 
        prior_preservation_class_folder = f"{self.cwd}/storage/{self.concept_name}/class_images" 
        class_data_root=prior_preservation_class_folder

        args = Namespace(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                resolution=resolution,
                center_crop=True,
                instance_data_dir=instance_data_dir,
                instance_prompt=ins_prompt,
                learning_rate=5e-06,
                max_train_steps=400,
                train_batch_size=1,
                gradient_accumulation_steps=2,
                max_grad_norm=1.0,
                mixed_precision="no", # set to "fp16" for mixed-precision training.
                gradient_checkpointing=True, # set this to True to lower the memory usage.
                use_8bit_adam=True, # use 8bit optimizer from bitsandbytes
                seed=3434554,
                with_prior_preservation=prior_preservation, 
                prior_loss_weight=prior_loss_weight,
                sample_batch_size=2,
                class_data_dir=prior_preservation_class_folder, 
                class_prompt=prior_preservation_class_prompt, 
                num_class_images=num_class_images, 
                output_dir=output_dir,
            )
        return args

    def load_pipe(self, model_name):
        #Load the Stable Diffusion model
        text_encoder = CLIPTextModel.from_pretrained(
            model_name, 
            subfolder="text_encoder", 
            use_auth_token=self.access_token
        )
        vae = AutoencoderKL.from_pretrained(
            model_name, 
            subfolder="vae", 
            use_auth_token=self.access_token
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_name, 
            subfolder="unet", 
            gradient_checkpointing=True,
            use_auth_token=self.access_token
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            use_auth_token=self.access_token,
        )
        return text_encoder, vae, unet, tokenizer

    def prior_preservation(self, args):
        args = vars(args)
        class_data_root = args['class_data_dir']
        pretrained_model_name_or_path = args['pretrained_model_name_or_path']
        class_prompt = args['class_prompt']
        num_class_images = args['num_class_images']

        #generate class images
        class_images_dir = Path(class_data_root)
        
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < num_class_images:
            pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path, 
                use_auth_token=self.access_token, 
                revision="fp16", 
                torch_dtype=torch.float16
            ).to("cuda")
            
            pipeline.enable_attention_slicing()
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = num_class_images - cur_class_images
            
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

    def record_model(self, model_dir):
        data =pd.read_csv(f"{self.cwd}/storage/data.csv")
        
        data.loc[len(data)] = tosave
        data.to_csv(f"{self.cwd}/storage/data.csv", index=False)
        return tosave

    def run_training(self, params):
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        #get and set all parameters
        args = self.set_params(params)

        #if prior preservation is True, make class images
        if vars(args)['with_prior_preservation']:
            self.prior_preservation(args)

        text_encoder, vae, unet, tokenizer = self.load_pipe(vars(args)["pretrained_model_name_or_path"])
        accelerate.notebook_launcher(training_function, args=(args, text_encoder, vae, unet, tokenizer), num_processes=1)
        
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        tosave = ["custom_model", self.concept_name, vars(args)["output_dir"]] 
        self.tracker.add_data(tosave)
        return tosave











