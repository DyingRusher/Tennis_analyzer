from ultralytics import YOLO
import cv2
import pickle

class PlayerTracker:

    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frames(self,frames,read_from_stubs=False,stub_path = None):
        player_dec = []

        if read_from_stubs and stub_path is not None:
            with open(stub_path,'rb') as f:
                player_dec = pickle.load(f)
            return player_dec
        
        for frame in frames:
            res = self.detect_frame(frame)
            player_dec.append(res)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(player_dec,f)
        return player_dec

    def detect_frame(self,frame):

        res = self.model.track(frame,persist=True)[0] 

        id_name_dict = res.names

        player_dict = {}
        for box in res.boxes:
            track_id = int(box.id.tolist()[0])
            re = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = re

        return player_dict
    def draw_box(self,video,detection):
        
        out_vi = []

        for i,(frame,player_dic) in enumerate(zip(video,detection)):
            if i > 50:
                break
            #draw player detection
            for track_id,box in player_dic.items():
                x1,y1,x2,y2 = box
                cv2.putText(frame,f"Player id:{track_id}",(int(box[0]),int(box[1]- 10)),cv2.FONT_HERSHEY_COMPLEX,0.6,(134,245,7),2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(10,213,123),2)
            out_vi.append(frame)
            # print(f'frame{i}/{len(video)}/{len(detection)}')

        return out_vi