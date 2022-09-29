import streamlit as st
import io
from PIL import Image
import os
import subprocess

st.write("""
Algovera Demo Stable Diffusion - Textual Inversion
""")

st.write("""
Step 1 - Train Textual Inversion
""")

uploaded_files = st.file_uploader("Upload images of ur concept",
                                 type=['jpg','jpeg','png'],
                                 help="Upload images of ur concept - jpg,jpeg,png", 
                                 accept_multiple_files=True,)

directory = "input_images"
if not os.path.exists(directory):
    os.mkdir(directory)

for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
     image = Image.open(io.BytesIO(bytes_data))
     image.save(f'{directory}/{uploaded_file.name}')

ins_prompt = st.text_input(label="instance_prompt", placeholder="`instance_prompt` is a prompt that should contain a good description of what your object or style is, together with the initializer word `sks`")
os.environ['INS_PROMPT'] = ins_prompt

prior_preservation_class_promp = st.text_input(label="prior preservation prompt", placeholder="`prior_preservation` option if you would like class of the concept (e.g.: toy, dog, painting) is guaranteed to be preserved. This increases the quality and helps with generalization at the cost of training time")
os.environ['PRIOR_PROMPT'] = prior_preservation_class_promp

def on_click():
    cmd = ['python', 'training.py']
    subprocess.run(cmd)

st.button(label="Run Training",
          on_click=on_click)