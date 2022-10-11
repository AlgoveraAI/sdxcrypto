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
from sdxcrypto.apps.utils.genimages import text2img, img2img
    
path_stg, path_ci, path_ai, path_ii = paths()
rmi_folder(path_ci)

def app():
    st.write("""
    # Generate Awesome Images
    """)

        
    col1, col2 = st.columns([4, 3], gap="small")

    with col1:
        if  len(glob.glob(f"{path_ci}/*jpg")) > 0:
            imgs = [load_image(fn) for fn in glob.glob(f"{path_ci}/*jpg")]

            for i, (img, fn) in enumerate(imgs):
                st.image(img)
                todelete = partial(delete, fn)
                st.button(label='Delete', key=f"Delete {i}", on_click=todelete)

        else:
            disp = st.image(Image.open('./sdxcrypto/placeholder.png'))

    with col2:
        tab1, tab2 = st.tabs(['text2img', 'img2img'])
        
        with tab1:
            text2img()

        with tab2:
            img2img()


    
    # if len(glob.glob(f"{path_ci}/*jpg")) > 0:
        
    #     st.markdown('''
    #     # Current Generated Images
    #     ''')

    #     for fn in glob.glob(f"{path_ci}/*jpg"):
    #         st.image(Image.open(fn))