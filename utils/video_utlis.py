import cv2

def read_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret,fram = cap.read()
        # print("he0",fram.shape)
        # fram = cv2.resize(fram,(244,244))
        if not ret:
            break
        frames.append(fram)
        
    cap.release()   
    return frames

def save_video(out_vi_frames,out_vi_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_vi_path,fourcc,24,(out_vi_frames[0].shape[1],out_vi_frames[0].shape[0]))
    for frame in out_vi_frames:
        out.write(frame)
    out.release()

