from utils import read_video,save_video
from trackers import PlayerTracker
def main():
    #convert video into list of frame
    in_video_path = 'tennis_video.mp4'
    video_frames = read_video(in_video_path)
    print(len(video_frames))
    #detect players from list of frame
    player_tra = PlayerTracker('yolov8l.pt')
    player_detection = player_tra.detect_frames(video_frames,True,stub_path='tracker_stubs/player_detction.pkl')

    #Draw output
    ##player
    # print("hey")
    out_vi = player_tra.draw_box(video_frames,player_detection)

    save_video(out_vi, 'output/output.avi')
    print("aj")

if __name__ == "__main__":
    main()
