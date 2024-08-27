from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np

class BallTracker:

    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolation_ball_position(self,ball_pos):  # give position of when it is undected by yolo

        ball_pos = [x.get(1,[]) for x in ball_pos] # x.get(key,value) if key is not present return value
        
        df_ball = pd.DataFrame(ball_pos,columns=['x1','y1','x2','y2'])
        
        #interpolation the missing
        df_ball = df_ball.interpolate() # give avg(around that frame) value to missing value
        
        df_ball = df_ball.bfill() # if first few frame has missing value initlize that value to first non-missing value
        
        #convert back from df to list
        df_ball = [{1:x} for x in df_ball.to_numpy().tolist()]
        
        return df_ball

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

    def get_ball_shot_frame(self,ball_pos):

        
            

        ball_pos = [x.get(1,[]) for x in ball_pos] # x.get(key,value) if key is not present return value       
        df_ball = pd.DataFrame(ball_pos,columns=['x1','y1','x2','y2'])
        
        df_ball['y_mid'] = (df_ball['y1'] + df_ball['y2'])/2
        df_ball['y_mid_rolling_mean'] = df_ball['y_mid'].rolling(window=5,min_periods= 1 , center=False).mean() # it will do mean of preivous 4 + current data 
        df_ball['delta_y'] = df_ball['y_mid_rolling_mean'].diff()
        df_ball['ball_hit'] = 0

        min_change_frame_for_hit = 25
        for i in range(1,len(df_ball) - int(min_change_frame_for_hit*1.2)):
            neg_position_change = df_ball['delta_y'].iloc[i] > 0 and df_ball['delta_y'].iloc[i+1] < 0
            pos_position_change = df_ball['delta_y'].iloc[i] < 0 and df_ball['delta_y'].iloc[i+1] > 0

            if neg_position_change or pos_position_change:
                change_count = 0
                for change_frame in range(i+1,i+int(min_change_frame_for_hit*1.2) +1):
                    neg_position_change_following_frame = df_ball['delta_y'].iloc[i] > 0 and df_ball['delta_y'].iloc[change_frame] < 0
                    pos_position_change_following_frame = df_ball['delta_y'].iloc[i] < 0 and df_ball['delta_y'].iloc[change_frame] > 0

                    if neg_position_change and neg_position_change_following_frame:
                        change_count += 1
                    elif pos_position_change and pos_position_change_following_frame:
                        change_count += 1

                if change_count > min_change_frame_for_hit -1 :
                    df_ball['ball_hit'].iloc[i] = 1

        frames_ball_hit= df_ball[df_ball['ball_hit'] == 1].index.tolist()
        return  frames_ball_hit
        

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
  
            #draw player detection
            for track_id,box in ball_dic.items():
                x1,y1,x2,y2 = box
                cv2.putText(frame,f"Ball id:{track_id}",(int(box[0]),int(box[1]- 10)),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,100,210),2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(20,101,231),2)
            out_vi.append(frame)
            # print(f'frame{i}/{len(video)}/{len(detection)}')

        return out_vi