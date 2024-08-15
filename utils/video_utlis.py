import cv2

def read_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    co = 0
    while co < 100:
        ret,fram = cap.read()
        if not ret:
            break
        frames.append(fram)
        co = co +1
    cap.release()
    return frames

def save_video(out_vi_frames,out_vi_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_vi_path,fourcc,24,(out_vi_frames[0].shape[1],out_vi_frames[0].shape[0]))
    for frame in out_vi_frames:
        out.write(frame)
    out.release()

