import os
import io
import glob
import json
import shutil
import zipfile
import requests 
from PIL import Image
from pathlib import Path
import streamlit as st

def paths():
    cwd = os.getcwd()
    path_stg = f"{cwd}/storage"                 # storage
    path_ci =  f"{cwd}/storage/current_images"  # current images
    path_ai =  f"{cwd}/storage/all_images"      # all images
    path_ii =  f"{cwd}/storage/init_images"     # initial image 

    return path_stg, path_ci, path_ai, path_ii

def build_folders():    
    path_stg, path_ci, path_ai, path_ii = paths()

    if not os.path.exists(path_stg):
        os.mkdir(path_stg)
    
    if not os.path.exists(path_ci):
        os.mkdir(path_ci)

    if not os.path.exists(path_ai):
        os.mkdir(path_ai)

    if not os.path.exists(path_ii):
        os.mkdir(path_ii)

    return path_stg, path_ci, path_ai, path_ii

def resize_saveimg(init_image, size=512):
    path_stg, path_ci, path_ai, path_ii = paths()

    bytes_data = init_image.read()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    image = image.resize((512, 512))
    
    image.save(f'{path_ii}/{init_image.name}')

def set_parameters(
    prompt,
    base_model,
    num_samples,
    height,
    width,
    inf_steps,
    guidance_scale,
    seed,
    strength
):
    os.environ["PROMPT"] = prompt
    os.environ["BASE_MODEL"] = base_model
    os.environ["NUM_SAMPLES"] = str(num_samples)
    os.environ["HEIGHT"] = str(height)
    os.environ["WIDTH"] = str(width)
    os.environ["INF_STEPS"] = str(inf_steps)
    os.environ["GUIDANCE_SCALE"] = str(guidance_scale)
    os.environ["SEED"] = str(seed)
    os.environ["STRENGTH"] = str(strength)

    parameters = {
            "base_model":os.getenv("BASE_MODEL"),
            "prompt":os.getenv("PROMPT"),
            "num_samples":int(os.getenv("NUM_SAMPLES")),
            "inf_steps": int(os.getenv("INF_STEPS")),
            "guidance_scale":float(os.getenv("GUIDANCE_SCALE")),
            "height":int(os.getenv("HEIGHT")),
            "width":int(os.getenv("WIDTH")),
            "seed":int(os.getenv("SEED")),
            "strength":float(os.getenv("STRENGTH"))
        }
    return parameters

def rmi_folder(path):
    for fn in glob.glob(f"{path}/*"):
        os.remove(fn)

def unzip_save_response(response):
    if response:
        path_stg, path_ci, path_ai, path_ii = paths()
        
        with open(f"{path_ci}/images.zip", "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(f"{path_ci}/images.zip","r") as zip_ref:
            zip_ref.extractall(f"{path_ci}")
        
        #remove zip file
        os.remove(f"{path_ci}/images.zip")
        
        for fn in glob.glob(f"{path_ci}/app/storage/output_images/*"):
            shutil.copy(fn, path_ai)
            shutil.copy(fn, path_ci)

        shutil.rmtree(f'{path_ci}/app')

def gen_image(
    prompt,
    base_model,
    num_samples=1,
    height=512,
    width=512,
    inf_steps=50,
    guidance_scale=7,
    seed=69,
    strength=0.6,
    init_image=None
):
    url = "http://fastapi:5000/generate"
    path_stg, path_ci, path_ai, path_ii = paths()

    #set up parameters
    parameters =  set_parameters(prompt=prompt,
                                base_model=base_model,
                                num_samples=num_samples,
                                height=height,
                                width=width,
                                inf_steps=inf_steps,
                                guidance_scale=guidance_scale,
                                strength=strength,
                                seed=seed)

    #remove images from the current_images folder
    rmi_folder(path_ci)
    
    #set up request body
    if init_image:
        file = {'files': open(f'{path_ii}/{init_image.name}', 'rb')}
    else: 
        file = None
    
    data = {'data': json.dumps(parameters)}
    
    #send request
    response = requests.post(url=url, 
                                data=data, 
                                files=file)
    
    #remove init_images
    rmi_folder(path_ii)

    #save zipped response
    unzip_save_response(response)

def allow_but():
    return True

def load_image(fn):
    im = Image.open(fn)
    return im, fn

def delete(fn):
    path_stg, path_ci, path_ai, path_ii = paths()
    Path(fn).unlink(missing_ok=True)
    Path(f"{path_ai}/{Path(fn).name}").unlink(missing_ok=True)

# class DisplayImages:
#     def __init__(self, imgs):
#         self.num = 0s
#         self.total = len(imgs)
#         self.imgs = imgs

#     def next_(self):
#         if self.num+1 > self.total:
#             self.num = 0
#         else:
#             self.num += 1

#     def display(self):
#         return self.imgs[self.num][0]
