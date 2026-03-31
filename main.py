from utils.video_utils import read_video_as_frames, save_video
from tracker import Tracker
from dotenv import load_dotenv
import os
import sys





def main():
    load_dotenv('.env')
    
    BASE_DIR = os.getenv('BASE_DIR')
    ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
    MODEL_PATH = os.path.join(BASE_DIR, 'best_model', 'weights', 'best.pt')


    video_name = sys.argv[1]
    
    video_full_path = os.path.join(ASSETS_DIR, video_name)



    video_frames = read_video_as_frames(video_path=video_full_path)
    tracker = Tracker(model_path=MODEL_PATH)
    tracker.get_object_tracks(video_frames)



    save_video(video_frames, '')
if __name__ == "__main__":
    main()