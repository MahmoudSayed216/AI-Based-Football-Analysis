from utils.video_utils import read_video_as_frames, write_frames_as_video
# from tracker import Tracker
from dotenv import load_dotenv
# from team_assigner import TeamAssigner
# from player_ball_assigner import PlayerBallAssigner
# from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer
# from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os
import sys
# import numpy as np
from analyzer import Analyzer


def main():
    load_dotenv('.env')
    
    BASE_DIR = os.getenv('BASE_DIR')
    ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
    MODEL_PATH = os.path.join(BASE_DIR, 'best_model', 'weights', 'best.pt')
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "output.mp4")
    STUBS_DIR = os.path.join(BASE_DIR, "analyzer", "stubs")
    os.makedirs(name=STUBS_DIR, exist_ok=True)

    video_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    read_from_stubs = bool(int(sys.argv[3]))
    print("READ FROM STUBS: ", read_from_stubs)
    video_full_path = os.path.join(ASSETS_DIR, video_name)



    video_frames = read_video_as_frames(video_path=video_full_path)
    analyzer = Analyzer(batch_size = batch_size, model_path=MODEL_PATH, stubs_dir=STUBS_DIR, read_from_stubs=read_from_stubs)
    output_frames = analyzer.analize(video_frames)
    write_frames_as_video(output_frames, OUTPUT_VIDEO_PATH)



if __name__ == "__main__":
    main()