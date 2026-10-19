import pandas as pd
from pathlib import Path


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
"""