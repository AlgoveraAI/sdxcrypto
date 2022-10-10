import io
import os
import glob
import json
from pathlib import Path
from PIL import Image
import subprocess
import streamlit as st
import requests
from typing import List
from sdxcrypto.api.tracker import BaseModels 

def app():
    
    tracker = BaseModels()
    cwd = os.getcwd()

    st.write("""
    Algovera Demo Stable Diffusion - Textual Inversion
    """)
    base_model = st.selectbox(
        'Choose your base model',
        (tracker.base_models())
        )

    concept_name = st.text_input(label="Concept Name", placeholder="Give your concept a name for eg. Galen")
    
    base_dir = f"{cwd}/storage/"
    concept_dir = f"{base_dir}/{concept_name}"
    final_dir = f"{concept_dir}/input_images"

    ins_prompt = st.text_input(label="Instance Prompt", placeholder="eg. photos of sks Galen")
    st.caption("`instance_prompt` is a prompt that should contain a good description of what your object or style is, together with the initializer word `sks`.")
 
    resolution = st.slider(label="Resoultion to be trained", min_value=128, max_value=512, step=128)
 
    st.markdown('''---''')
    prior = st.radio(label="Train Priors", options=["Yes", "No"], index=1)

    if prior == "No":
        disabled=True
    else:
        disabled=False

    prior_prompt = st.text_input(label="Prior preservation prompt", disabled=disabled)
    st.caption ("`prior_preservation` option if you would like class of the concept (e.g.: toy, dog, painting) is guaranteed to be preserved. This increases the quality and helps with generalization at the cost of training time")
    st.markdown('''---''')
    
    uploaded_files = st.file_uploader("Upload images of ur concept",
                                    type=['jpg','jpeg','png'],
                                    help="Upload images of ur concept - jpg, jpeg, png", 
                                    accept_multiple_files=True,)

    input_dir = f"{cwd}/storage/{concept_name}/input_images"
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))
        image.save(f'{input_dir}/{uploaded_file.name}')

    def set_parameters(concept_name=concept_name, ins_prompt=ins_prompt, prior_prompt=prior_prompt):
        os.environ["BASE_MODEL"] = base_model
        os.environ["CONCEPT_NAME"] = concept_name
        os.environ["INS_PROMPT"] = ins_prompt
        os.environ["RESOLUTION"] = str(resolution)
        os.environ["PRIOR"] = prior
        os.environ["PRIOR_PROMPT"] = prior_prompt

    def run_training():
        url = "http://fastapi:5000/train"
        
        set_parameters()
        
        parameters = {
            "base_model": os.getenv("BASE_MODEL"),
            "concept_name": os.getenv("CONCEPT_NAME"),
            "ins_prompt": os.getenv("INS_PROMPT"),
            "resolution": int(os.getenv("RESOLUTION")),
            "prior": True if os.getenv("PRIOR") == "Yes" else False,
            "prior_prompt": os.getenv("PRIOR_PROMPT"),
        }

        data = {'data': json.dumps(parameters)}
    
        files = [("files", open(fn, "rb")) for fn in glob.glob(f"{input_dir}/*.jpg")]

        response = requests.post(url, data=data, files=files)
        res = response.json()
        
        toadd = [res['model_type'], res['model_name'], res['model_dir']]
        tracker.add_data(toadd)
        # tracker.del_data('b')

    st.button(label="Train", on_click=run_training)