from ultralytics import YOLO

yolo = YOLO('./runs/detect/train4/weights/best.pt')
valid_results = yolo.val(data="./data.yaml",imgsz=416)
print(valid_results)