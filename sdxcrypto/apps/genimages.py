import os
import io
import glob
import json
import zipfile
import requests 
import shutil
import streamlit as st
from PIL import Image
from sdxcrypto.api.tracker import BaseModels
from sdxcrypto.apps.utils import build_folders, resize_saveimg, set_parameters, rmi_folder, unzip_save_response
    
def app():
    st.write("""
    Generate Awesome Images
    """)
    bm = BaseModels()
    cwd = os.getcwd()
    
    path_stg, path_ci, path_ai, path_ii = build_folders()

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
        step=128,
        value=512
    )

    width = st.slider(
        label="Width of the generated image",
        min_value=128,
        max_value=512, 
        step=128,
        value=512
    )
    
    inf_steps = st.slider(
        label="Number of inference steps",
        min_value=10,
        max_value=100, 
        step=10,
        value=50
    )
    
    guidance_scale = st.slider(
        label="Guidance Scale",
        min_value=1,
        max_value=100, 
        step=1,
        value=7
    )

    seed = st.number_input(label="Seed", step=1, value=42)

    init_image = st.file_uploader("Upload initial image",
                                    type=['jpg','jpeg','png'],
                                    help="Upload an initial image - jpg, jpeg, png", 
                                    accept_multiple_files=False)

    if init_image:
        resize_saveimg(init_image, 512)

    def gen_image():
        url = "http://fastapi:5000/generate"
        
        #set up parameters
        parameters =  set_parameters(prompt,
                                    base_model,
                                    num_samples,
                                    height,
                                    width,
                                    inf_steps,
                                    guidance_scale,
                                    seed)

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
    
    st.button(label="Diffuse My Images", on_click=gen_image)

    for fn in glob.glob(f"{path_ci}/*jpg"):
        st.image(Image.open(fn))



