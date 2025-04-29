import os
import random
import shutil
from pathlib import Path

# ==== CONFIGURATION ====
dataset_dir = Path("/Users/tomaszfrelek/Downloads/Self Driving Car/export")  # Has images/ and labels/ folders
output_root = Path("/Users/tomaszfrelek/Downloads/partitioned_data")          # Output folder root
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
seed = 42

# ========================
random.seed(seed)

# Input folders
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

# Output folder setup
for split in ["train", "val", "test"]:
    (output_root / split / "images").mkdir(parents=True, exist_ok=True)
    (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

# Get and shuffle all images
image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
random.shuffle(image_paths)

# Calculate split sizes
total = len(image_paths)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_paths = image_paths[:train_end]
val_paths = image_paths[train_end:val_end]
test_paths = image_paths[val_end:]

# Helper function to copy matched image-label pairs
def copy_data(image_list, split_name):
    for img_path in image_list:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"⚠️  Label not found for: {img_path.name}, skipping.")
            continue
        shutil.copy2(img_path, output_root / split_name / "images" / img_path.name)
        shutil.copy2(label_path, output_root / split_name / "labels" / label_path.name)

# Copy files
copy_data(train_paths, "train")
copy_data(val_paths, "val")
copy_data(test_paths, "test")

print("Dataset successfully split into 80% train, 10% val, 10% test.")
