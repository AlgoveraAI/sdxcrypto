
import os
from PIL import Image
import numpy as np 
import glob
import streamlit as st 
from functools import partial
from sdxcrypto.apps.utils.utils import load_image, delete

# Function to Read and Manupilate Images
def app():
    cwd = os.getcwd()

    st.write('This is the home page of  `Alogvera\'s Stable Diffusion x Crypto` Project.')

    st.write('In this app, you can introduce new concepts into SD models (such as yourself) and generate images using your best prompts.')
    
    imgs = [load_image(fn) for fn in glob.glob(f"{cwd}/storage/all_images/*")]

    st.write('''
    # Your Gallery
    ''')

    # col1, col2, col3 = st.columns([3,3,3], gap="small")
    # for i, (img, fn) in enumerate(imgs):

    #     st.image(img)
    #     todelete = partial(delete, fn)
    #     st.button(label=f"Delete {i+1}", on_click=todelete)
    n_cols = 3
    n_rows = 1 + len(imgs) // int(n_cols)
    rows = [st.container() for _ in range(n_rows)]
    cols_per_row = [r.columns(n_cols) for r in rows]
    cols = [column for row in cols_per_row for column in row]

    for i, (img, fn) in enumerate(imgs):
        cols[i].image(img)
        todelete = partial(delete, fn)
        cols[i].button(label="Delete", key=f"Delete {i}", on_click=todelete)
        