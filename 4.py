from ultralytics import YOLO

model = YOLO(r"details_dataset\runs\detect\train\weights\best.pt")

model("details_1.mp4", save=True)