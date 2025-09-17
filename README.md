# DART Project

Scripts that combine **YOLO** object detection with **SAM (Segment Anything Model)** segmentation to detect and segment objects in images.


## Requirements

- Python 3.8+
- Libraries:
  - torch, torchvision
  - numpy
  - opencv-python
  - matplotlib
  - ultralytics (for YOLO)
  - segment-anything (for SAM)

Install with:

```bash
pip install torch torchvision
pip install numpy opencv-python matplotlib ultralytics
pip install git+https://github.com/facebookresearch/segment-anything.git
```

To clone this repo:
```
git clone git@github.com:cando-jo/dart-yolo-sam.git
cd dart-yolo-sam
```


## Scripts Overview

- **copy_train_images.py**  
  Copies matching images (jpg/png/jpeg) from the implants dataset into the YOLO training folder, based on existing label files. Warns if a label has no matching image.

- **train_yolo.py**  
  Trains a YOLOv8 model (default: `yolov8n.pt`) on the implant dataset using `dataset.yaml`. Saves results and weights (e.g. `best.pt`) into the `yolo_results` folder.

- **yolo_sam_segmentation.py**  
  Runs inference with trained YOLO + SAM:  
  - Detects implants with YOLO.  
  - Segments them with SAM.  
  - Saves mask images.  
  - Exports a COCO-style `annotations.json` with bounding boxes + segmentations.  

