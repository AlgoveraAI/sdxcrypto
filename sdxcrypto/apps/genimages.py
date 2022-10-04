import os
import requests 
import streamlit as st
from sdxcrypto.api.tracker import BaseModels


def app():
    st.write("""
    Generate Awesome Images
    """)
    bm = BaseModels()

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

    def gen_image():
        url = "http://localhost:3333/generate"
        
        parameters = {
            "base_model":os.getenv("BASE_MODEL"),
            "prompt":os.getenv("PROMPT"),
            "num_samples":os.getenv("NUM_SAMPLES"),
            "inf_steps": os.getenv("INF_STEPS"),
            "guidance_scale":os.getenv("GUIDANCE_SCALE"),
            "height":os.getenv("HEIGHT"),
            "width":os.getenv("WIDTH"),
            "seed":os.getenv("SEED")

        }

        response = requests.post(url=url, params=parameters)

    st.button(label="Diffuse My Images", on_click=gen_image)