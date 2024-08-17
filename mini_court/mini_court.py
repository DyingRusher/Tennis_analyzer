import cv2
import sys
import constants
from utils import pixal_to_m,meter_to_p

class Mini_court:

    def __init__(self,frame):
        self.rectangle_width = 250
        self.rectangle_height = 450
        self.buffer = 50
        self.padding = 20


        self.set_background_rectangle(frame)
        self.set_mini_court()


    def m_to_p(self,dis):
        return int(meter_to_p(dis,constants.DOUBLE_LINE_WIDTH,self.c_rectangle_width))
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
        # drawing_keypoints[8] = 



    def set_mini_court(self):
        self.c_start_x = self.start_x + self.padding
        self.c_start_y = self.start_y + self.padding
        self.c_end_x = self.end_x - self.padding
        self.c_end_y = self.end_y - self.padding
        self.c_rectangle_width = self.c_end_x - self.c_start_x 

    def set_background_rectangle(self,frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.rectangle_height
        self.start_x = self.end_x - self.rectangle_width
        self.start_y = self.end_y - self.rectangle_height