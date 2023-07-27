from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
model = YOLO('yolov8n.yaml')

# Train the model
model.train(data='data.yaml', epochs=1000, imgsz=640)

# For Prediction
model.predict('C:\\Users\\archa\\Desktop\\C-DAC\\Using_YoloV8(6 classes)\\Maize Quality Detection.v3i.yolov8\\test', save = True)


