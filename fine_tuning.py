from ultralytics import YOLO

yolo = YOLO('yolo12m.pt')
yolo.train(data='./data.yaml', epochs=40, save=True,imgsz=416,cache=True, amp=False)
valid_results = yolo.val()
print(valid_results)
yolo.export()