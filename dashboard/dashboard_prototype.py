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
import pandas as pd
import streamlit as st
import numpy as np 
from PIL import Image, UnidentifiedImageError
import json

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Report Directories
EVAL_DIR = BASE_DIR / "reports" / "evaluation_"
RECALL_DIR = BASE_DIR / "reports" / "recall_optimization"

# Allow Python to import from project root
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Model inference
from inference.predict import load_model, predict_image

# Load the trained model
@st.cache_resource
def get_model():
    """
    Load the trained EfficientNet-B0 model.

    Streamlit caches the model to avoid reloading
    the weights whenever the dashboard refreshes.
    """

    return load_model()

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


# Upload Imaging here, will later
# implement the use of the model.


with tab1:

    st.header("Image Classification")

    uploaded_file = st.file_uploader(
        "Upload a dermoscopic image",
        type=["jpg", "jpeg", "png"],
        max_upload_size=10
    )

    # Check if an image has been uploaded
    if uploaded_file is not None:
        try:
            raw = uploaded_file.getvalue()

            with Image.open(BytesIO(raw)) as img:
                image = np.array(img.convert("RGB"))

            # -------------------------------------------
            # IMAGE PREPROCESSING

            processed = preprocess_image(image.copy())

            # Apply deterministic evaluation transformations
            transforms = get_transforms(training=False)

            image_tensor = transforms(
                image=processed
            )["image"]

            # Convert tensor back to displayable RGB values
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            preview = image_tensor.permute(
                1, 2, 0
            ).numpy()

            preview = np.clip(preview * std + mean, 0, 1)

            # Display original and preprocessed images
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption="Original Image",
                    width="stretch"
                )

            with col2:
                st.image(
                    preview,
                    caption="Preprocessed Image",
                    width="stretch"
                )

            st.success("Image Preprocessing Successful")

            # -------------------------------------------
            # MODEL PREDICTION
            # -------------------------------------------

            st.subheader("Model Prediction")

            if st.button("Classify Image", type="primary"):

                with st.spinner("Analyzing skin lesion..."):

                    # Load trained EfficientNet-B0
                    model = get_model()

                    # Run model inference
                    result = predict_image(
                        model,
                        BytesIO(raw)
                    )

                # -------------------------------------------
                # PREDICTION RESULTS

                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Class",
                        result["class"].upper()
                    )

                with col2:
                    st.metric(
                        "Model Probability",
                        f"{result['confidence']:.2%}"
                    )

                # -------------------------------------------
                # CLASS PROBABILITIES

                st.subheader("Class Probabilities")

                probabilities_df = pd.DataFrame(
                    [
                        {
                            "Class": class_name.upper(),
                            "Probability": probability
                        }
                        for class_name, probability
                        in result["probabilities"].items()
                    ]
                )

                probabilities_df = probabilities_df.sort_values(
                    by="Probability",
                    ascending=False
                )

                st.bar_chart(
                    probabilities_df.set_index("Class")
                )

                st.caption(
                    "These are uncalibrated model probabilities, "
                    "not clinical diagnostic confidence."
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ) as error:
            st.error(f"Unable to process image: {error}")

# -------------------------------------------
# TAB 2 Model Metrics


# Shows the model performance:

# Recall : For all the positives, how many did they find.
# Precision : For all the positives, How many were actually positive
# F1-Score : Balance of both percision and recall

# Recall = TP / TP + FN
# Precision = TP / TP + FP

# Melonoma Recall, Melanoma
# BCC Recall, Basal Cell Carcinoma
# AKIEC Recall, Actinic Keratoses and Intraepithelial carcinoma

# Confusition Matrix, where the model made mistakes

with tab2:

    st.header("Model Performance")
    st.caption("Evaluation results from the saved reports")

    # -------------------------------------------
    # LOAD EVALUATION RESULTS

    with open(EVAL_DIR / "summary.json", "r") as file:
        summary = json.load(file)

    per_class_df = pd.read_csv(
        EVAL_DIR / "per_class_metrics.csv"
    )

    # -------------------------------------------
    # HIGH-RISK CLASS RECALL

    st.subheader("High-Risk Lesion Recall")

    recall_by_class = per_class_df.set_index("class")["recall"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Melanoma Recall",
        f"{recall_by_class['mel']:.2%}"
    )

    col2.metric(
        "BCC Recall",
        f"{recall_by_class['bcc']:.2%}"
    )

    col3.metric(
        "AKIEC Recall",
        f"{recall_by_class['akiec']:.2%}"
    )

    # -------------------------------------------
    # OVERALL MODEL PERFORMANCE

    st.subheader("Overall Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Macro F1",
        f"{summary['macro_f1']:.2%}"
    )

    col2.metric(
        "Weighted F1",
        f"{summary['weighted_f1']:.2%}"
    )

    col3.metric(
        "Test Images",
        f"{summary['n_test_images']:,}"
    )

    # -------------------------------------------
    # CONFUSION MATRIX

    st.subheader("Confusion Matrices")

    st.image(
        str(EVAL_DIR / "confusion_matrices.png"),
        width="stretch"
    )

    # -------------------------------------------
    # PER-CLASS PERFORMANCE

    st.subheader("Per-Class Performance")

    display_df = per_class_df.copy()

    for column in [
        "precision",
        "recall",
        "recall_ci_low",
        "recall_ci_high",
        "f1"
    ]:
        display_df[column] = (
            display_df[column] * 100
        ).round(2)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

    # -------------------------------------------
    # ADDITIONAL EVALUATION

    with st.expander("ROC and Precision-Recall Curves"):
        st.image(
            str(EVAL_DIR / "curves.png"),
            width="stretch"
        )

    with st.expander("Model Calibration"):
        st.image(
            str(EVAL_DIR / "calibration.png"),
            width="stretch"
        )

        st.write(
            "ECE Before Calibration:",
            f"{summary['ece_before']:.4f}"
        )

        st.write(
            "ECE After Calibration:",
            f"{summary['ece_after']:.4f}"
        )

    with st.expander("High-Risk Operating Point"):

        high_risk = summary[
            "high_risk_at_validation_90_specificity"
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Recall",
            f"{high_risk['recall']:.2%}"
        )

        col2.metric(
            "Specificity",
            f"{high_risk['specificity']:.2%}"
        )

        col3.metric(
            "Precision",
            f"{high_risk['precision']:.2%}"
        )

    st.caption(
        "Evaluation results may aggregate multiple folds. "
        "The live classifier uses best_model_fold_5.pth. "
        "Performance metrics are not clinical guarantees."
    )

# -------------------------------------------
# TAB 3 Model Card

#Documentation for the Project, 
#A Technical Information Sheet
with tab3:
    st.header("Model Information")

    st.subheader("Dataset")
    st.write("HAM10000 dermoscopic image dataset.")

    st.subheader("Architecture")
    st.write("EfficientNet-B0 (PyTorch / torchvision)")

    st.subheader("Model Checkpoint")
    st.write("best_model_fold_5.pth")

    st.subheader("Input Image Size")
    st.write("224 × 224 pixels")

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