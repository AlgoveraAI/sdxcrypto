import os
import io
import glob
import json
import shutil
import zipfile
import requests 
from PIL import Image

def build_folders():
    cwd = os.getcwd()
    path_stg = f"{cwd}/storage"                 # storage
    path_ci =  f"{cwd}/storage/current_images"  # current images
    path_ai =  f"{cwd}/storage/all_images"      # all images
    path_ii =  f"{cwd}/storage/init_images"     # initial image 
    
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
    path_stg, path_ci, path_ai, path_ii = build_folders()

    bytes_data = init_image.read()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    image = image.resize((768, 512))
    
    image.save(f'{path_ii}/{init_image.name}')

def set_parameters(
    prompt,
    base_model,
    num_samples,
    height,
    width,
    inf_steps,
    guidance_scale,
    seed
):
    os.environ["PROMPT"] = prompt
    os.environ["BASE_MODEL"] = base_model
    os.environ["NUM_SAMPLES"] = str(num_samples)
    os.environ["HEIGHT"] = str(height)
    os.environ["WIDTH"] = str(width)
    os.environ["INF_STEPS"] = str(inf_steps)
    os.environ["GUIDANCE_SCALE"] = str(guidance_scale)
    os.environ["SEED"] = str(seed)

    parameters = {
            "base_model":os.getenv("BASE_MODEL"),
            "prompt":os.getenv("PROMPT"),
            "num_samples":int(os.getenv("NUM_SAMPLES")),
            "inf_steps": int(os.getenv("INF_STEPS")),
            "guidance_scale":float(os.getenv("GUIDANCE_SCALE")),
            "height":int(os.getenv("HEIGHT")),
            "width":int(os.getenv("WIDTH")),
            "seed":int(os.getenv("SEED"))
        }
    return parameters

def rmi_folder(path):
    for fn in glob.glob(f"{path}/*"):
        os.remove(fn)

def unzip_save_response(response):
    if response:
        path_stg, path_ci, path_ai, path_ii = build_folders()
        
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