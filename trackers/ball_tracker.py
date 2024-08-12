from ultralytics import YOLO
import cv2
import pickle

class BallTracker:

    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frames(self,frames,read_from_stubs=False,stub_path = None):
        ball_dec = []

        if read_from_stubs and stub_path is not None:
            with open(stub_path,'rb') as f:
                ball_dec = pickle.load(f)
            return ball_dec
        
        for frame in frames:
            res = self.detect_frame(frame)
            ball_dec.append(res)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(ball_dec,f)
        return ball_dec

    def detect_frame(self,frame):

        res = self.model.predict(frame)[0] 

        id_name_dict = res.names

        ball_dict = {}
        for box in res.boxes:

            re = box.xyxy.tolist()[0]    
            ball_dict[1] = re

        return ball_dict
    def draw_box(self,video,detection):
        
        out_vi = []

        for i,(frame,ball_dic) in enumerate(zip(video,detection)):
            if i > 50:
                break   
            #draw player detection
            for track_id,box in ball_dic.items():
                x1,y1,x2,y2 = box
                cv2.putText(frame,f"Ball id:{track_id}",(int(box[0]),int(box[1]- 10)),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,100,210),2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(20,101,231),2)
            out_vi.append(frame)
            # print(f'frame{i}/{len(video)}/{len(detection)}')

        return out_vi