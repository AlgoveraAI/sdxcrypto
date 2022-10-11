import os
import io
import glob
import json
import zipfile
import requests 
import shutil
import streamlit as st
from PIL import Image
from functools import partial
from sdxcrypto.api.tracker import BaseModels
from sdxcrypto.apps.utils.utils import (build_folders, resize_saveimg, set_parameters, 
                                  rmi_folder, unzip_save_response, gen_image,
                                  allow_but, load_image, paths, delete)


def text2img():
    bm = BaseModels()
    cwd = os.getcwd()
    
    path_stg, path_ci, path_ai, path_ii = build_folders()
    
    base_model = st.selectbox(
            'Choose your model',
            (bm.base_models()),
            key=1
    )

    prompt = st.text_input(label="Prompt", 
        placeholder="Whats your prompt",
        on_change=allow_but,
        key=2
    )

    num_samples = st.slider(
        label="Number of Samples to Generate",
        min_value=1,
        max_value=3,
        key=3
    )

    height = st.slider(
        label="Height of the generated image",
        min_value=128,
        max_value=512, 
        step=128,
        value=512,
        key=4
    )

    width = st.slider(
        label="Width of the generated image",
        min_value=128,
        max_value=512, 
        step=128,
        value=512,
        key=5
    )
    
    inf_steps = st.slider(
        label="Number of inference steps",
        min_value=10,
        max_value=100, 
        step=10,
        value=50,
        key=6
    )
    
    guidance_scale = st.slider(
        label="Guidance Scale",
        min_value=1,
        max_value=100, 
        step=1,
        value=7,
        key=7
    )

    seed = st.number_input(
        label="Seed", 
        step=1, 
        value=42,
        key=8
    )

    # init_image = st.file_uploader("Upload initial image",
    #                                 type=['jpg','jpeg','png'],
    #                                 help="Upload an initial image - jpg, jpeg, png", 
    #                                 accept_multiple_files=False)

    # if init_image:
    #     resize_saveimg(init_image, 512)
    
    gen_image_partial = partial(gen_image, 
                        prompt=prompt,
                        base_model=base_model,
                        num_samples=num_samples,
                        height=height,
                        width=width,
                        inf_steps=inf_steps,
                        guidance_scale=guidance_scale,
                        seed=seed
                        )
    st.button(label="Diffuse My Images", 
              on_click=gen_image_partial,
              key=16
            )

def img2img():
    bm = BaseModels()
    cwd = os.getcwd()
    
    path_stg, path_ci, path_ai, path_ii = build_folders()
    
    base_model = st.selectbox(
            'Choose your model',
            (bm.base_models()),
            key=9
    )

    prompt = st.text_input(label="Prompt", 
        placeholder="Whats your prompt",
        on_change=allow_but,
        key=10
    )

    num_samples = st.slider(
        label="Number of Samples to Generate",
        min_value=1,
        max_value=3, 
        key=11
    )
    
    inf_steps = st.slider(
        label="Number of inference steps",
        min_value=10,
        max_value=100, 
        step=10,
        value=50,
        key=12
    )
    
    guidance_scale = st.slider(
        label="Guidance Scale",
        min_value=1,
        max_value=100, 
        step=1,
        value=7,
        key=13
    )

    strength = st.slider(
        label="Strength",
        min_value=0.,
        max_value=1., 
        step=0.1,
        value=0.8,
    )

    seed = st.number_input(
        label="Seed", 
        step=1, 
        value=42,
        key=14
    )

    init_image = st.file_uploader("Upload initial image",
                                    type=['jpg','jpeg','png'],
                                    help="Upload an initial image - jpg, jpeg, png", 
                                    accept_multiple_files=False,
                                    key=15
                                )

    if init_image:
        resize_saveimg(init_image, 512)

    gen_image_partial = partial(gen_image, 
                        prompt=prompt,
                        base_model=base_model,
                        num_samples=num_samples,
                        inf_steps=inf_steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        init_image=init_image,
                        strength=strength
                    )

    st.button(label="Diffuse My Images", 
              on_click=gen_image_partial,
              key=17
            )