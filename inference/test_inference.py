"""
test_inference.py

Josh Navarro

Goal: Testing the inference functions for
predict.py

Run:
python -m inference.test_inference
"""

# Imports
from pathlib import Path
from inference.predict import (
    load_model,
    predict_image
)

# Load trained model
model = load_model()

# Find an image to test
PROJECT_ROOT = Path(__file__).resolve().parent.parent

image_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "HAM10000_IMG_P1"
    / "ISIC_0024306.jpg"
)

if not image_path.is_file():
    raise FileNotFoundError(f"Test image not found: {image_path}")

# Generate prediction
result = predict_image(model, image_path)

print("Predicted Class:", result["class"])
print("Model Probability:", f"{result['confidence']:.2%}")

print("\nAll Class Probabilities:")

for class_name, probability in result["probabilities"].items():
    print(f"{class_name}: {probability:.2%}")


"""
output:
Training and evaluation transforms created!
Predicted Class: nv
Model Probability: 99.97%

All Class Probabilities:
nv: 99.97%
mel: 0.01%
bkl: 0.01%
bcc: 0.00%
akiec: 0.00%
vasc: 0.00%
df: 0.00%
"""