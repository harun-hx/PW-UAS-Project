import kagglehub
import os
import shutil
import random
from pathlib import Path

# --- Configuration ---
# MAPPING based on dataset metadata
BREED_MAPPING = {
    "01": "Akhal-Teke",
    "02": "Appaloosa",
    "03": "Orlov Trotter",
    "04": "Vladimir Heavy Draft",
    "05": "Percheron",
    "06": "Arabian",
    "07": "Friesian"
}

# Your target destination
BASE_DIR = r"C:\Users\Harun\Documents\Web\ProjectNateHiggerson\horse-dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# Split ratio (20% for validation)
VAL_SPLIT = 0.2 

def setup_directories():
    """Creates the empty train/val folder structure."""
    # We create folders for every breed found in the mapping
    for breed in BREED_MAPPING.values():
        os.makedirs(os.path.join(TRAIN_DIR, breed), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, breed), exist_ok=True)
    print(f"Created directories in {BASE_DIR}")

def organize_dataset(source_path):
    source_path = Path(source_path)
    print("\nProcessing images...")

    # Recursively find all images (handles if they are nested deeper)
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        all_images.extend(source_path.rglob(ext))

    print(f"Found {len(all_images)} total images. Sorting them now...")

    counters = {breed: 0 for breed in BREED_MAPPING.values()}

    for img_path in all_images:
        filename = img_path.name
        
        # Check the prefix (e.g., "01_123.png")
        prefix = filename.split('_')[0]
        
        if prefix in BREED_MAPPING:
            breed_name = BREED_MAPPING[prefix]
            
            # Decide: Train or Val?
            is_val = random.random() < VAL_SPLIT
            target_folder = VAL_DIR if is_val else TRAIN_DIR
            
            # Copy the file
            shutil.copy2(img_path, os.path.join(target_folder, breed_name, filename))
            counters[breed_name] += 1
        else:
            # print(f"Skipping unknown file: {filename}")
            pass

    print("\nSummary of organized images:")
    for breed, count in counters.items():
        print(f"  > {breed}: {count} images")

if __name__ == "__main__":
    # 1. Download
    print("Step 1: Downloading dataset from Kaggle...")
    download_path = kagglehub.dataset_download("olgabelitskaya/horse-breeds")
    print(f"Downloaded to cache: {download_path}")

    # 2. Create your folder structure
    print("\nStep 2: Creating folder structure...")
    setup_directories()

    # 3. Move and Split
    print("\nStep 3: Organizing and splitting data...")
    organize_dataset(download_path)
    
    print("\n✅ Done! Your dataset is ready.")