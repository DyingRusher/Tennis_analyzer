from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetection

def main():

    #convert video into list of frame
    in_video_path = 'tennis_video.mp4'
    video_frames = read_video(in_video_path)

    #detect players from list of frame
    player_tra = PlayerTracker('yolov8l.pt')
    ball_tra = BallTracker('best3.pt')
    #court line dect
    co_line_tra = CourtLineDetection('court_line.pth')


    ball_detection = ball_tra.detect_frames(video_frames,True,stub_path='tracker_stubs/ball_detction.pkl')
    player_detection = player_tra.detect_frames(video_frames,True,stub_path='tracker_stubs/player_detction.pkl')
    court_keypoints = co_line_tra.predict(video_frames[0])

    #Draw output
    ##player
    # print(len(player_detection),len(ball_detection))

    out_vi = ball_tra.draw_box(video_frames,ball_detection)
    out_vi = player_tra.draw_box(out_vi,player_detection)
    out_vi = co_line_tra.draw_kp_video(out_vi,court_keypoints)


    save_video(out_vi, 'output/output.avi')
    # save_video(out_vi2, 'output/output2.avi')
    # save_video(out_vi, 'output/output_ball.avi')
    print("aj")

if __name__ == "__main__":
    main()
