"""
predict.py

Josh Navarro

Goal: Using the trained model, predict
the skin lesion image
"""

# Imports
# -----------------------------------

import torch 
import torch.nn as nn
from pathlib import Path
from torchvision import models

# Import prev prepreocessing funcs
from preprocessing.preprocessing import prepare_inference_image

# Configurations
# -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_fold_2.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Class ordering from the training notebook
CLASS_NAMES = [
    "nv",
    "mel",
    "bkl",
    "bcc",
    "akiec",
    "vasc",
    "df"
]

# Model
# -----------------------------------
def create_model():
    """
    Create the EfficientNet-B0 architecture.
    """

    # Weights = none, using our pretrained weights
    model = models.efficientnet_b0(weights=None)

    # Match the seven output classes used during training
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        len(CLASS_NAMES)
    )

    return model

def load_model(model_path=MODEL_PATH):
    """
    Load the trained EfficientNet-B0 model.
    """

    # Validate the model path
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    # Init the model
    model = create_model()

    # Load checkpoint from trusted source
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model

# Predictions
# -----------------------------------


def predict_image(model, image_path):
    """
    Predict the skin lesion class using the trained
    EfficientNet-B0 model.

    Args:
        model: Loaded EfficientNet-B0 model.
        image_path: Path or file-like object containing
                    the uploaded image.

    Returns:
        Dictionary containing the predicted class,
        probability, and probabilities for all classes.
    """

    # Step 1: Reuse the existing preprocessing pipeline
    image_tensor = prepare_inference_image(image_path)

    # Step 2: Move image tensor to the model's device
    image_tensor = image_tensor.to(DEVICE)

    # Step 3: Run model inference
    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    # Step 4: Convert probabilities to CPU
    probabilities = probabilities[0].cpu()

    # Step 5: Find predicted class
    predicted_idx = probabilities.argmax().item()

    # Step 6: Return prediction results
    return {
        "class": CLASS_NAMES[predicted_idx],
        "confidence": probabilities[predicted_idx].item(),
        "probabilities": {
            class_name: probabilities[i].item()
            for i, class_name in enumerate(CLASS_NAMES)
        }
    }
