"""
F-08 for the Project

Status : Protoype #1, No Model Implementation

Run:
python -m streamlit run dashboard_prototype.py

"""
# Imports
from pathlib import Path
import sys
from io import BytesIO

import streamlit as st
import numpy as np 
from PIL import Image, UnidentifiedImageError

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Allow Python to import from project root
sys.path.insert(0, str(BASE_DIR))

# Fundtions/Var from preprocessing file
from preprocessing.preprocessing import (
    CONFIG,
    preprocess_image,
    get_transforms
)

# Dashboard Configuation
st.set_page_config(
    page_title="Skin Lesion Classifier",
    layout="wide"
)

# Start of the Dashboard/Ui
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

"""
Upload Imaging here, will later
implement the use of the model.
"""
with tab1:

    st.header("Image Classification")

    uploaded_file = st.file_uploader(
        "Upload a demoscopic image",
        type = ["jpg", "jpeg", "png"],
        max_upload_size = 10
    )

    # Checks if there something in the file
    if uploaded_file is not None:
        try:
            raw = uploaded_file.getvalue()

            with Image.open(BytesIO(raw)) as img:
                image = np.array(img.convert("RGB"))

            processed = preprocess_image(
                image.copy(),
                CONFIG
            )

            # apply the deterministic evaluation transformation
            transforms = get_transforms(
                CONFIG, training = False
            )

            image_tensor = transforms(
                image = processed
            )["image"]

            # Convert this back into displayable RGB Values
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224,0.225])

            preview = image_tensor.permute(
                1, 2, 0
            ).numpy()

            preview = np.clip(preview * std + mean, 0, 1)

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption = "Original Image",
                    width = "stretch"
                )
            with col2:
                st.image(
                    preview,
                    caption = "Preprocessed Image",
                    width = "stretch"
                )
            st.sucess("Image Preprocessing Sucessful")
            st.subheader("Model Prediction")
            st.info(
                "Prediction functionality will be enabled "
                "once the trained model is available."
            )
        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ) as error:
            st.error(f"Unable to process image: {error}")
# -------------------------------------------
# TAB 2 Model Metrics
"""
Shows the model performance:

Recall : For all the positives, how many did they find.
Precision : For all the positives, How many were actually positive
F1-Score : Balance of both percision and recall

Recall = TP / TP + FN
Precision = TP / TP + FP

Melonoma Recall, Melanoma
BCC Recall, Basal Cell Carcinoma
AKIEC Recall, Actinic Keratoses and Intraepithelial carcinoma

Confusition Matrix, where the model made mistakes
"""
with tab2:
    st.header("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Melanoma Recall", "N/A")
    col2.metric("BCC Recall", "N/A")
    col3.metric("AKIEC Recall", "N/A")

    st.subheader("Confusion Matrix")
    st.info("Awaiting model evaluation results.")

    st.subheader("Per-Class Performance")
    st.info(
        "Precision, recall, and F1 scores will appear here."
    )


# -------------------------------------------
# TAB 3 Model Card
"""
Documentation for the Project, 
A Technical Information Sheet
"""
with tab3:
    st.header("Model Information")

    st.subheader("Dataset")
    st.write("HAM10000 dermoscopic image dataset.")

    st.subheader("Architecture")
    st.write("EfficientNet-B0 (planned initial backbone).")

    st.subheader("Classification Categories")
    st.write(
        "nv, mel, bkl, bcc, akiec, vasc, df"
    )

    st.subheader("Intended Use")
    st.write(
        "Research and exploratory analysis of "
        "dermoscopic skin lesion images."
    )

    st.subheader("Limitations")
    st.write(
        "The dataset contains class imbalance and "
        "representation gaps. Performance on smartphone "
        "or other non-dermoscopic images is unvalidated. "
        "The model is not approved for clinical diagnosis."
    )