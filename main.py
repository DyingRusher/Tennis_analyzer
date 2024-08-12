from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker
def main():

    #convert video into list of frame
    in_video_path = 'tennis_video.mp4'
    video_frames = read_video(in_video_path)

    #detect players from list of frame
    player_tra = PlayerTracker('yolov8l.pt')

    ball_tra = BallTracker('best3.pt')

    ball_detection = ball_tra.detect_frames(video_frames,True,stub_path='tracker_stubs/ball_detction.pkl')
    player_detection = player_tra.detect_frames(video_frames,True,stub_path='tracker_stubs/player_detction.pkl')

    #Draw output
    ##player
    # print(len(player_detection),len(ball_detection))

    out_vi = ball_tra.draw_box(video_frames,ball_detection)
    out_vi2 = player_tra.draw_box(video_frames,player_detection)

    save_video(out_vi, 'output/output.avi')
    # save_video(out_vi2, 'output/output2.avi')
    # save_video(out_vi, 'output/output_ball.avi')
    print("aj")

if __name__ == "__main__":
    main()
