from utils.video_utils import read_video_as_frames, write_frames_as_video
from tracker import Tracker
from dotenv import load_dotenv
import os
import sys




def main():
    load_dotenv('.env')
    
    BASE_DIR = os.getenv('BASE_DIR')
    ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
    MODEL_PATH = os.path.join(BASE_DIR, 'best_model', 'weights', 'best.pt')
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "output.mp4")
    cur_dir_name = os.path.dirname(os.path.abspath(__file__))
    STUBS_DIR = os.path.join(cur_dir_name, "stubs")
    os.makedirs(name=STUBS_DIR, exist_ok=True)

    video_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    read_from_stubs = bool(int(sys.argv[3]))
    video_full_path = os.path.join(ASSETS_DIR, video_name)



    video_frames = read_video_as_frames(video_path=video_full_path)

    tracker = Tracker(model_path=MODEL_PATH, batch_size=batch_size)

    tracks = tracker.get_object_tracks(video_frames, read_from_stubs = read_from_stubs, stubs_path=STUBS_DIR+'/stubs.pkl')

    output_frames = tracker.draw_annotations(video_frames, tracks)


    write_frames_as_video(output_frames, OUTPUT_VIDEO_PATH)



if __name__ == "__main__":
    main()