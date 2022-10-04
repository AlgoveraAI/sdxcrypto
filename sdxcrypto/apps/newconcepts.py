import io
import os
from PIL import Image
import subprocess
import streamlit as st
from .utils.tracker import BaseModels 
from .utils.training import Training
def app():
    
    bm = BaseModels()
    trn = Training()

    st.write("""
    Algovera Demo Stable Diffusion - Textual Inversion
    """)
    model = st.selectbox(
        'Choose your base model',
        (bm.base_models())
        )

    concept_name = st.text_input(label="Concept Name", placeholder="Give your concept a name for eg. Galen")

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
                                    help="Upload images of ur concept - jpg,jpeg,png", 
                                    accept_multiple_files=True,)

    directory = f"storage/{concept_name}"
    if not os.path.exists(directory):
        os.mkdir(directory)
        os.mkdir(f"{directory}/input_images")

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        image = Image.open(io.BytesIO(bytes_data))
        image.save(f'{directory}/input_images/{uploaded_file.name}')

    def set_parameters(concept_name=concept_name, ins_prompt=ins_prompt, prior_prompt=prior_prompt):
        os.environ["MODEL"] = model
        os.environ["CONCEPT_NAME"] = concept_name
        os.environ["INS_PROMPT"] = ins_prompt
        os.environ["RESOLUTION"] = str(resolution)
        os.environ["PRIOR"] = prior
        os.environ["PRIOR_PROMPT"] = prior_prompt

    def run_training():
        set_parameters()
        trn.run_training()
        
    st.button(label="Train", on_click=run_training)