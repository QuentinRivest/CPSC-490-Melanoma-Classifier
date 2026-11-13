# Melanoma Classifier

### Contributors :

- Josh Navarro
- Quentin Rivest
- Nathan Koh
- Albert Razo
- Nelson Vargas

### Overview :
IMPORTANT: please load HAM10000_metadata, HAM10000_images_part_1, and HAM10000_images_part_2 to data/raw/ on your own machine
so github doesn't have to store multiple gigabytes of image files. I listed data/raw/ in the .gitignore so it shouldn't
be a problem.

## Features 5 and 6

- `recall_optimization/Feature_5.ipynb`: Calibrates model scores and
  selects recall-focused thresholds using validation data.
- `evaluation/Feature_6.ipynb`: Generates test metrics, confidence
  intervals, confusion matrices, and ROC/PR curves.

**To run:** Open the notebooks in Colab and run Feature 5 before
Feature 6. No GPU is required.

Both use `real_predictions.csv` in the shared Drive folder:
`MyDrive/MelanomaClassfierProject/Features_5_6`.
Make this folder accessible through your My Drive and ensure you
have Editor access to save results.

Report snapshots are under `reports`. BCC and AKIEC meet their
recall/specificity targets; melanoma recall is 76.9%, below 85%.
These results use threshold flags, not top-1 classification.

### Framework :
- Pytorch

### Summary :


