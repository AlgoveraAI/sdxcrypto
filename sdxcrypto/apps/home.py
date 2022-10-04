
from PIL import Image
import numpy as np 
import glob
import streamlit as st 

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    return im

def app():
    st.title('Home')

    st.write('This is the home page of  `Alogvera\'s Stable Diffusion x Crypto` Project.')

    st.write('In this app, you can introduce new concepts into SD models (such as yourself) and generate images using your best prompts.')
    
    imgs = [load_image(fn) for fn in glob.glob("storage/output_images/*")]

    for img in imgs:
        st.image(img)
