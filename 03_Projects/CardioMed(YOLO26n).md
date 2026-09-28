from ultralytics import YOLO  
  
model = YOLO("yolo26n.pt")  
  
results = model.train(data = r"C:\Users\happy\Documents\Беловольская Ольга, ИСвГС и СИЯКИСРО, презентации и прочее\ИСвГС\ИСвГС (311-22). Проекты, презентации и прочее\Для диплома\MI-ECG\data.yaml", epochs = 300, imgsz = 640, patience = 200)  
  
metrics = model.val(data = r"C:\Users\happy\Documents\Беловольская Ольга, ИСвГС и СИЯКИСРО, презентации и прочее\ИСвГС\ИСвГС (311-22). Проекты, презентации и прочее\Для диплома\MI-ECG\data.yaml", imgsz = 640)