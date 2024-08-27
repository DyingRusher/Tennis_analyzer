import cv2
import sys
import constants
from utils import pixal_to_m,meter_to_p
import numpy as np
class Mini_court:

    def __init__(self,frame):
        self.rectangle_width = 250
        self.rectangle_height = 500
        self.buffer = 50
        self.padding = 20


        self.set_background_rectangle(frame)
        self.set_mini_court()
        self.set_c_keypoints()
        self.set_c_lines()


    def m_to_p(self,dis):
        return int(meter_to_p(dis,constants.DOUBLE_LINE_WIDTH,self.c_rectangle_width))
    

    def set_c_lines(self):
        self.lines = [
            (0,2),(4,5),(6,7),(1,3),(0,1),(8,9),(10,11),(12,13),(2,3)
        ]
        # pass
    def set_c_keypoints(self):
        drawing_keypoints = [0]*28

        #point 1
        drawing_keypoints[0] , drawing_keypoints[1] = int(self.c_start_x) , int(self.c_start_y)
        #p2
        drawing_keypoints[2] , drawing_keypoints[3] = int(self.c_end_x) , int(self.c_start_y)
        #p3
        drawing_keypoints[4]  = int(self.c_start_x)
        #we are writing very long code to we make helper function
        drawing_keypoints[5] = self.c_start_y + int(meter_to_p(constants.HALF_COURT_LINE_HEIGHT*2,constants.DOUBLE_LINE_WIDTH,self.c_rectangle_width))
        #p4
        drawing_keypoints[6] = self.c_start_x + self.c_rectangle_width
        drawing_keypoints[7] = drawing_keypoints[5]
        #p5
        drawing_keypoints[8] = drawing_keypoints[0] + self.m_to_p(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[9] = drawing_keypoints[1]
        #p6
        drawing_keypoints[10] = drawing_keypoints[4] + self.m_to_p(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[11] = drawing_keypoints[5]
        #p7
        drawing_keypoints[12] = drawing_keypoints[2] - self.m_to_p(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[13] = drawing_keypoints[3]
        #p8
        drawing_keypoints[14] = drawing_keypoints[6] - self.m_to_p(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_keypoints[15] = drawing_keypoints[7]
        #p9
        drawing_keypoints[16] = drawing_keypoints[8]
        drawing_keypoints[17] = drawing_keypoints[9] + self.m_to_p(constants.NO_MANS_LAND_HEIGHT)
        #p10
        drawing_keypoints[18] = drawing_keypoints[16] + self.m_to_p(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17]
        #p11
        drawing_keypoints[20] = drawing_keypoints[10]
        drawing_keypoints[21] = drawing_keypoints[11] - self.m_to_p(constants.NO_MANS_LAND_HEIGHT)
        #p12
        drawing_keypoints[22] = drawing_keypoints[20] + self.m_to_p(constants.SINGLE_LINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21] 
        #p13
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18])/2)
        drawing_keypoints[25] = drawing_keypoints[17] 
        #p14
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22])/2)
        drawing_keypoints[27] = drawing_keypoints[21] 

        self.drawing_keypoints = drawing_keypoints

    def set_mini_court(self):
        self.c_start_x = self.start_x + self.padding
        self.c_start_y = self.start_y + self.padding
        self.c_end_x = self.end_x - self.padding
        self.c_end_y = self.end_y - self.padding
        self.c_rectangle_width = self.c_end_x - self.c_start_x
        # print("asd",self.c_end_x,self.c_start_x)

    def set_background_rectangle(self,frame):
        
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.rectangle_height
        self.start_x = self.end_x - self.rectangle_width
        self.start_y = self.end_y - self.rectangle_height

    def draw_back_rect(self,frame):
        # below is code for doing transparency copy of img1 on main img
        shapes = np.zeros_like(frame,np.uint8) 
        cv2.rectangle(shapes,(self.start_x,self.start_y),(self.end_x,self.end_y),(255,255,255),-1)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame,alpha,shapes,1 - alpha,0)[mask] # out = frame*alpha + shapes*(1 - alpha ) + 0
        return out
    
    def draw_court(self,frame):
        for i in range(0,len(self.drawing_keypoints),2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame,(x,y),5,(250,0,0),-1)

        #draw lines
        for line in self.lines:
            start_p = (int(self.drawing_keypoints[line[0]*2]),int(self.drawing_keypoints[line[0]*2 +1]))
            end_p = (int(self.drawing_keypoints[line[1]*2]),int(self.drawing_keypoints[line[1]*2+1]))
            cv2.line(frame,start_p,end_p,(0,0,0),2)
        

        #draw net line
        cv2.line(frame,(self.c_start_x,int((self.c_start_y+self.c_end_y) /2)),(self.c_end_x,int((self.c_start_y+self.c_end_y) /2)),(0,0,123),2)
        
        return frame
    
    def draw_mini_court(self,frames):
        out = []
        for frame in frames:
            frame = self.draw_back_rect(frame)
            frame = self.draw_court(frame)  
            out.append(frame)
        return out
    
    def get_start_p_c(self):
        return (self.c_start_x,self.c_start_y)
    
    def get_c_width(self):
        return self.c_rectangle_width
    
    def get_c_keypoints(self):
        return self.drawing_keypoints
