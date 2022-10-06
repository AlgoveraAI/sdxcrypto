
import os
from PIL import Image
import numpy as np 
import glob
import streamlit as st 
from functools import partial

# Function to Read and Manupilate Images
def load_image(fn):
    im = Image.open(fn)
    return im, fn

def app():
    cwd = os.getcwd()

    st.write('This is the home page of  `Alogvera\'s Stable Diffusion x Crypto` Project.')

    st.write('In this app, you can introduce new concepts into SD models (such as yourself) and generate images using your best prompts.')
    
    imgs = [load_image(fn) for fn in glob.glob(f"{cwd}/storage/all_images/*")]

    st.write('''
    Your Gallery
    ''')

    def delete(fn):
        os.remove(fn)

    for i, (img, fn) in enumerate(imgs):
        st.image(img)
        todelete = partial(delete, fn)
        st.button(label=f"Delete {i+1}", on_click=todelete)
