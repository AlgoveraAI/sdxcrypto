import os
import streamlit as st
from .utils.tracker import BaseModels 
from .utils.inference import Inference 

def app():
    bm = BaseModels()
    inf = Inference()

    st.write("""
    Generate Awesome Images
    """)

    option = st.selectbox(
        'Choose your model',
        (bm.base_models()))

    prompt = st.text_input(label="Prompt", 
        placeholder="Whats your prompt")

    num_samples = st.slider(
        label="Number of Samples to Generate",
        min_value=1,
        max_value=5, 
    )
    def set_parameters(option=option, prompt=prompt, num_samples=num_samples):
        os.environ["PROMPT"] = prompt
        os.environ["OPTION"] = option
        os.environ["NUM_SAMPLES"] = str(num_samples)

    def gen_image():
        set_parameters()
        inf.run_inference()

    st.button(label="Diffuse My Images", on_click=gen_image)