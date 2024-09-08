from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetection
from mini_court import Mini_court
import cv2

def main():

    #convert video into list of frame
    in_video_path = 'tennis_video.mp4'
    video_frames = read_video(in_video_path)
    #Mini court
    mini_court = Mini_court(video_frames[0])
    
    #detect players from list of frame
    player_tra = PlayerTracker('yolov8l.pt')
    ball_tra = BallTracker('best3.pt')
    
    #court line dect
    co_line_tra = CourtLineDetection('keypoint_model.pth')

    ball_detection = ball_tra.detect_frames(video_frames,True,stub_path='tracker_stubs/ball_detction.pkl')
    ball_detection = ball_tra.interpolation_ball_position(ball_detection)

    
    # print(ball_detection,ball_detection2)
    player_detection = player_tra.detect_frames(video_frames,True,stub_path='tracker_stubs/player_detction.pkl')
    court_keypoints = co_line_tra.predict(video_frames[0])

    #get ball shot frame
    ball_shot_frame = ball_tra.get_ball_shot_frame(ball_detection)
    print(ball_shot_frame)

    # only select two person which are close to court
    player_detection = player_tra.filter_player(player_detection,court_keypoints)

    # get keypoints of player and ball
    player_mini_court_dec , ball_mini_court_dec = mini_court.convert_bbox_to_mini_court(player_detection,ball_detection,court_keypoints)
    
    #Draw output
    out_vi = ball_tra.draw_box(video_frames,ball_detection)
    out_vi = player_tra.draw_box(out_vi,player_detection)
    
    out_vi = mini_court.draw_point_on_mini_court(out_vi,player_mini_court_dec)
    out_vi = mini_court.draw_point_on_mini_court(out_vi,ball_mini_court_dec,color = (0,255,123))

    out_vi = co_line_tra.draw_kp_video(out_vi,court_keypoints)
    # print(len(out_vi))

    # draw mini court
    out_vi = mini_court.draw_mini_court(out_vi)
    # print(len(out_vi))

    ## draw frame number top right corner
    for i,frame in enumerate(out_vi):
     
        cv2.rectangle(frame,(40,10),(640,120),(230,150,100),-1)
        cv2.putText(frame,f"Frame:{i}",(50,100),cv2.FONT_HERSHEY_SIMPLEX,4,(50,10,18),4)

    save_video(out_vi, 'output/output.avi')
    print("aj")

if __name__ == "__main__":
    main()
