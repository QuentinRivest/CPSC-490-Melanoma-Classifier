import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


# HELPER FUNCTION TO RESOLVE IMAGE PATH

IMAGE_DIRS = [
    Path("data/raw/HAM10000_images_part_1"),
    Path("data/raw/HAM10000_images_part_2")
]

def find_image(image_id):
    for directory in IMAGE_DIRS:
        candidate = directory / f"{image_id}.jpg"

        if candidate.exists():
            return str(candidate)

    return None


# HELPER FUNCTION FOR COUNTING DX

def get_class_counts(df):
    return(df["dx"].value_counts().sort_index().to_dict())


print('\n' + '-'*30 + '\n')


print("Reading metadata")
df = pd.read_csv("data/raw/HAM10000_metadata")


# Verify there are no missing images

print("Verifying dataset integrity")
df["image_path"] = df["image_id"].apply(find_image)
missing = df[df["image_path"].isnull()]
print(f"Missing images: {len(missing)}")
assert len(missing) == 0, "Dataset integrity violation"


print('\n' + '-'*30 + '\n')


# Number of each type of lesion

print("Counting dx")
dx_counts = df["dx"].value_counts()
with open("reports/dx_counts.txt", "w") as report:
    report.write("Lesion Type Count Report\n")
    report.write("========================\n\n")
    
    for dx, count in dx_counts.items():
        report.write(f"{dx:6}: {count:4} images\n")
print("Report saved to reports/dx_counts.txt")


# Number of images corresponding to a given lesion

print("Counting duplicate lesions")
lesion_counts = df["lesion_id"].value_counts()

duplicated_lesions = lesion_counts[lesion_counts > 1]

duplicated_df = duplicated_lesions.reset_index()
duplicated_df.columns = ["lesion_id", "image_count"]

duplicated_df.to_csv(
    "data/duplicate_lesions.csv",
    index=False
)

with open("reports/duplicate_lesions.txt", "w") as report:
    report.write("Duplicate Lesions Report\n")
    report.write(f"{len(duplicated_lesions)} lesions have multiple images\n\n")
    report.write("========================\n\n")

    for lesion_id, count in duplicated_lesions.items():
        report.write(f"{lesion_id}: {count} images\n")

print("Report saved to data/duplicate_lesions.txt")


print('\n' + '-'*30 + '\n')


print("Creating folds")
skgf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

fold_statistics = {
    "metadata": {
        "n_folds": 5,
        "splitter": "StratifiedGroupKFold",
        "group_column": "lesion_id",
        "target_column": "dx"
    },
    "folds": {}
}

X = df
y = df["dx"]
groups = df["lesion_id"]

for i, (train_index, test_index) in enumerate(skgf.split(X, y, groups), start=1):
    print(f"\nWorking Fold {i}")
    
    outer_train_df = df.iloc[train_index]
    print("Training set built")
    
    test_df = df.iloc[test_index]
    print("Test set built")

    print("Verifying no leakage")
    train_lesions = set(outer_train_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    assert(train_lesions.isdisjoint(test_lesions))
    print("No leakage found")

    print(f"Carving 15% of test split as a validation split")
    inner_skgf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42) # 7 splits is closest to each split being 15% of the set
    inner_train_index, val_index = next(inner_skgf.split(outer_train_df, outer_train_df["dx"], outer_train_df["lesion_id"]))
    train_df = outer_train_df.iloc[inner_train_index]
    val_df = outer_train_df.iloc[val_index]

    fold_dir = Path(f"data/folds/fold_{i}")
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving splits for Fold {i}")
    print(f"Creating {fold_dir}/train.csv")
    train_df.to_csv(fold_dir / "train.csv", index=False)
    print(f"Creating {fold_dir}/val.csv")
    val_df.to_csv(fold_dir / "val.csv", index=False)
    print(f"Creating {fold_dir}/test.csv")
    test_df.to_csv(fold_dir / "test.csv", index=False)

    fold_statistics["folds"][f"fold_{i}"] = {
        "train": get_class_counts(train_df),
        "val": get_class_counts(val_df),
        "test": get_class_counts(test_df)
    }


print('\n' + '-'*30 + '\n')


print("Reporting per-fold per-split dx counts")
with open("reports/fold_statistics.json", "w") as report:
    json.dump(fold_statistics, report, indent=4)
print("Saved to reports/fold_statistics.json")

print('\n' + '-'*30 + '\n')


print("Data ingestion complete")


"""
# //////////////////////////////////////////////////////
# OLD IMPLEMENTATION USING 70/15/15 train/val/test SPLIT
# //////////////////////////////////////////////////////

lesion_df = df.groupby("lesion_id").first().reset_index()

X = lesion_df["lesion_id"]
y = lesion_df["dx"]


# Seperate 15% test, 85% train and validation

train_val_ids, test_ids = train_test_split(
    X,
    test_size=0.15,
    stratify=y,
    random_state=42
)

train_val_df = lesion_df[lesion_df["lesion_id"].isin(train_val_ids)]


# Separate 15% (of the original dataset) validation and 70% (of the original dataset) test

train_ids, val_ids = train_test_split(
    train_val_df["lesion_id"],
    test_size=(15/85),
    stratify=train_val_df["dx"],
    random_state=42
)

train_df = df[df["lesion_id"].isin(train_ids)]
val_df = df[df["lesion_id"].isin(val_ids)]
test_df = df[df["lesion_id"].isin(test_ids)]

# Verify no leakage between splits

train_set = set(train_df["lesion_id"])
val_set = set(val_df["lesion_id"])
test_set = set(test_df["lesion_id"])

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

train_df.to_csv(
    "data/splits/train.csv",
    index=False
)

val_df.to_csv(
    "data/splits/val.csv",
    index=False
)

test_df.to_csv(
    "data/splits/test.csv",
    index=False
)
"""