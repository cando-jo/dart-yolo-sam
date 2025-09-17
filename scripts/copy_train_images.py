import os
import shutil

# Paths
labels_folder = "C:\qusai_playground\DART\DART\YOLO\project_yolo\labels\train"
implants_folder = "C:\OAIData\implants"
train_images_folder = "C:\qusai_playground\DART\DART\YOLO\project_yolo\images\train"

# Create train folder if it doesn't exist
os.makedirs(train_images_folder, exist_ok=True)

# Get all label filenames without extension
label_basenames = [os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(".txt")]

copied_count = 0

for base in label_basenames:
    found = False
    # Check for common image extensions
    for ext in [".jpg", ".png", ".jpeg"]:
        img_path = os.path.join(implants_folder, base + ext)
        if os.path.exists(img_path):
            shutil.copy(img_path, train_images_folder)
            copied_count += 1
            found = True
            break
    if not found:
        print(f"Warning: No image found for label {base}")

print(f"Copied {copied_count} images to {train_images_folder}")
