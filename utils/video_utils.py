import cv2
import matplotlib.pyplot as plt


def read_video_as_frames(video_path):
    print("reading video")
    cap = cv2.VideoCapture(filename = video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    return frames








def write_frames_as_video(output_video_frames, save_path):
    print("saving output video")
    height, width, _ = output_video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 25.0, (width, height))
    for frame in output_video_frames:
        frame = cv2.cvtColor(frame, code=cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()

