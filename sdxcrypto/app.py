import streamlit as st
from multiapp import MultiApp
from sdxcrypto.apps import home, genimages, newconcepts # import your app modules here

app = MultiApp()

st.markdown("""
# Stable Diffusion x Crypto
Your creative assistant. Take your writings to the next level.
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Generate Images", genimages.app)
app.add_app("Introduce Concepts", newconcepts.app)
# The main app
app.run()