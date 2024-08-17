from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_dis,find_center_bbox
class PlayerTracker:

    def __init__(self,model_path):
        self.model = YOLO(model_path)


    def filter_player(self,player_detection,court_detection):
        player_detection_1_frame = player_detection[0]
        # print("c",court_detection)

        chosen_player = self.choose_player(player_detection_1_frame,court_detection)
        #this logic is not applicable in this video so i did it manually
        chosen_player = (1,2)
        # print('f',chosen_player)
        final_player = []
        for i in player_detection:
            final_player_dict = {track_id:bbox for track_id,bbox in i.items() if track_id in chosen_player}
            final_player.append(final_player_dict)
        
        
        return final_player
    
    def choose_player(self,player_dict,court_dect):
        dis = []

        #below code does keep record of every [player id , nearest point to court in entire video] and appendind to dis
        for track_id,bbox in player_dict.items():
            center = find_center_bbox(bbox)
            
            min_dis = float('inf')
            for i in range(0,len(court_dect),2):
                court_keypoint = (court_dect[i],court_dect[i+1])
                di = measure_dis(center,court_keypoint)
                # print(track_id,i,di,court_keypoint,center)

                if di < min_dis:
                    min_dis = di
                
            dis.append([track_id,min_dis])
        
        # sort to minimum distance to court
        dis.sort(key = lambda x : x[1])
        # print(dis)
        return (dis[0][0],dis[1][0])

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
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(134,245,7),2)
            out_vi.append(frame)
            # print(f'frame{i}/{len(video)}/{len(detection)}')

        return out_vi