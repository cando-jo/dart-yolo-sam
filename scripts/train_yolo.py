import sys
print(sys.executable)


from ultralytics import YOLO

# Create a YOLO model (pretrained yolov8n weights)
model = YOLO("yolov8n.pt")  # you can change to yolov8s.pt or yolov8m.pt for larger models
# Train the model
model.train(
    data="DART\scripts\dataset.yaml",
    epochs=50,          # adjust depending on convergence
    imgsz=640,          # image size
    batch=4,            # batch size
    project="C:\qusai_playground\DART\DART\YOLO\project_yolo\yolo_results",
    name="implant_detector",
    exist_ok=True       # overwrite if folder exists
)
