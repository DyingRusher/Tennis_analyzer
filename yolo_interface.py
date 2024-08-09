from ultralytics import YOLO

model = YOLO('best3.pt')
model.predict('tennis_video.mp4',conf = 0.4,save=True)