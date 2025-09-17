import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# ============================
# CONFIGURATION
# ============================

YOLO_WEIGHTS = r"C:\qusai_playground\DART\DART\YOLO\project_yolo\yolo_results\implant_detector\weights\best.pt"
IMAGES_FOLDER = r"C:\qusai_playground\DART\DART\YOLO\project_yolo\images\val"
SAM_CHECKPOINT = r"C:\yolo_sam\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
OUTPUT_FOLDER = r"C:\qusai_playground\DART\DART\YOLO\project_yolo\sam_output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
MASKS_FOLDER = os.path.join(OUTPUT_FOLDER, "masks")
os.makedirs(MASKS_FOLDER, exist_ok=True)
JSON_PATH = os.path.join(OUTPUT_FOLDER, "annotations.json")

# ============================
# LOAD MODELS
# ============================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_WEIGHTS)

print("Loading SAM model...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device)
sam_predictor = SamPredictor(sam)

# ============================
# Prepare JSON structure
# ============================

coco_json = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "implant"}],
}
annotation_id = 1

# ============================
# Process images
# ============================

image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_id, img_file in enumerate(tqdm(image_files, desc="Processing images")):
    img_path = os.path.join(IMAGES_FOLDER, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Failed to load image: {img_path}")
        continue

    h, w = image.shape[:2]

    # YOLO detection
    results = yolo_model(img_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

    if len(boxes) == 0:
        print(f"⚠️ No YOLO detections for {img_file}")
        continue

    # SAM setup
    sam_predictor.set_image(image)

    # Add image info to JSON
    coco_json["images"].append({
        "id": img_id,
        "file_name": img_file,
        "height": h,
        "width": w
    })

    # Loop over detected boxes
    for box_idx, box in enumerate(boxes):
        # SAM prediction with multimask_output=True
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=True
        )

        # Pick mask with largest area
        areas = [mask.sum() for mask in masks]
        if len(areas) == 0 or max(areas) == 0:
            print(f"⚠️ SAM failed to segment box {box_idx+1} in {img_file}")
            continue

        mask = masks[np.argmax(areas)].astype("uint8") * 255

        # Save mask image
        mask_filename = f"{os.path.splitext(img_file)[0]}_mask{box_idx+1}.png"
        cv2.imwrite(os.path.join(MASKS_FOLDER, mask_filename), mask)

        # Convert mask to polygon for COCO JSON
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for cnt in contours:
            cnt = cnt.flatten().tolist()
            if len(cnt) >= 6:  # at least 3 points
                segmentation.append(cnt)

        # Compute bounding box
        x, y, w_box, h_box = cv2.boundingRect(mask)

        # Add annotation to JSON
        coco_json["annotations"].append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": segmentation,
            "bbox": [x, y, w_box, h_box],
            "iscrowd": 0
        })
        annotation_id += 1

# ============================
# Save JSON
# ============================
with open(JSON_PATH, "w") as f:
    json.dump(coco_json, f, indent=4)

print("✅ Masks and COCO JSON saved to:", OUTPUT_FOLDER)
