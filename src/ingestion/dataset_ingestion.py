import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


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


df = pd.read_csv("data/raw/HAM10000_metadata")


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


# Seperate 15% (of the original dataset) validation and 70% (of the original dataset) test

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



# ///////////////////////////////////////////////////////////////////
# EVERYTHING BELOW THIS WAS PRETTY MUCH JUST FOR STATISTICS REPORTING
# FEEL FREE TO IGNORE OR DELETE IF WE DEEM IT UNNECESSARY
# ///////////////////////////////////////////////////////////////////


"""
# Verify there are no missing images
 
df["image_path"] = df["image_id"].apply(find_image)
missing = df[df["image_path"].isnull()]
print(f"Missing images: {len(missing)}")


# Number of each type of lesion

dx_counts = df["dx"].value_counts()
with open("reports/dx_counts.txt", "w") as report:
    report.write("Lesion Type Count Report\n")
    report.write("========================\n\n")
    
    for dx, count in dx_counts.items():
        report.write(f"{dx:6}: {count:4} images\n")


# Number of images corresponding to a given lesion

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


# dx proportions should be relatively consistent across splits

with open("reports/per-class_per-split_image_counts.txt", "w") as report:
    report.write("Per-Class, Per-Split Image Count Proportions Report\n")
    report.write("===================================================\n\n")
    report.write("TRAINING SET\n")
    report.write(str(train_df["dx"].value_counts(normalize=True)) + "\n\n")
    report.write("\nVALIDATION SET\n")
    report.write(str(val_df["dx"].value_counts(normalize=True)) + "\n\n")
    report.write("\nTEST SET\n")
    report.write(str(test_df["dx"].value_counts(normalize=True)))
"""