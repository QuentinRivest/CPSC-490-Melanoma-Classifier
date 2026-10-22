import streamlit as st
import numpy as np 
from PIL import Image
from preprocessing import (
    CONFIG,
    preprocess_image,
    get_transforms
)

# Dashboard Configuation
st.set_page_config(
    page_title="Skin Lesion Classifier",
    layout="wide"
)

st.title("Skin Lesion Classification")
st.caption("HAM10000 | Machine Learning Research Prototype")

st.warning(
    "Research prototype only. This application is not "
    "intended for medical diagnosis."
)

tab1, tab2, tab3 = st.tabs([
    "Classifier",
    "Model Metrics",
    "Model Card"
])

# -------------------------------------------
# TAB 1 Classifier

# -------------------------------------------
# TAB 2 Model Metrics

# -------------------------------------------
# TAB 3 Model Card
