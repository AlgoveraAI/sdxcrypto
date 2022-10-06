import os
import io
import glob
import zipfile
import requests 
import shutil
import streamlit as st
from PIL import Image
from sdxcrypto.api.tracker import BaseModels


def build_folders():
    cwd = os.getcwd()
    if not os.path.exists(f"{cwd}/storage"):
        os.mkdir(f"{cwd}/storage")
    
    if not os.path.exists(f"{cwd}/storage/current_images"):
        os.mkdir(f"{cwd}/storage/current_images")

    if not os.path.exists(f"{cwd}/storage/all_images"):
        os.mkdir(f"{cwd}/storage/all_images")
    
def app():
    st.write("""
    Generate Awesome Images
    """)
    bm = BaseModels()
    cwd = os.getcwd()
    
    build_folders()
    
    path_ci = f"{cwd}/storage/current_images"
    path_ai = f"{cwd}/storage/all_images"

    base_model = st.selectbox(
        'Choose your model',
        (bm.base_models()))

    prompt = st.text_input(label="Prompt", 
        placeholder="Whats your prompt")

    num_samples = st.slider(
        label="Number of Samples to Generate",
        min_value=1,
        max_value=5, 
    )
    height = st.slider(
        label="Height of the generated image",
        min_value=128,
        max_value=512, 
        step=128
    )

    width = st.slider(
        label="Width of the generated image",
        min_value=128,
        max_value=512, 
        step=128
    )
    
    inf_steps = st.slider(
        label="Number of inference steps",
        min_value=10,
        max_value=100, 
        step=10
    )
    
    guidance_scale = st.slider(
        label="Guidance Scale",
        min_value=1,
        max_value=100, 
        step=1
    )

    seed = st.number_input(label="seed", step=1)

    def set_parameters():
        os.environ["PROMPT"] = prompt
        os.environ["BASE_MODEL"] = base_model
        os.environ["NUM_SAMPLES"] = str(num_samples)
        os.environ["HEIGHT"] = str(height)
        os.environ["WIDTH"] = str(width)
        os.environ["INF_STEPS"] = str(inf_steps)
        os.environ["GUIDANCE_SCALE"] = str(guidance_scale)
        os.environ["SEED"] = str(seed)

        print(prompt, base_model, num_samples, height, width, inf_steps, guidance_scale, seed)

    def gen_image():
        url = "http://fastapi:5000/generate"
        
        set_parameters()

        for fn in glob.glob(f"{path_ci}/*jpg"):
            os.remove(fn)
        
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

        response = requests.post(url=url, json=parameters)
        
        with open(f"{path_ci}/images.zip", "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(f"{path_ci}/images.zip","r") as zip_ref:
            zip_ref.extractall(f"{path_ci}")
        
        os.remove(f"{path_ci}/images.zip")
        
        for fn in glob.glob(f"{path_ci}/app/storage/output_images/*"):
            shutil.copy(fn, path_ai)
            shutil.copy(fn, path_ci)
        shutil.rmtree(f'{path_ci}/app')
    
    st.button(label="Diffuse My Images", on_click=gen_image)

    for fn in glob.glob(f"{path_ci}/*.jpg"):
        st.image(Image.open(fn))



