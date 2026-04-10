from utils.video_utils import read_video_as_frames, write_frames_as_video
from tracker import Tracker
from dotenv import load_dotenv
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os
import sys
import numpy as np



def main():
    load_dotenv('.env')
    
    BASE_DIR = os.getenv('BASE_DIR')
    ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
    MODEL_PATH = os.path.join(BASE_DIR, 'best_model', 'weights', 'best.pt')
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "output.mp4")
    STUBS_DIR = os.path.join(BASE_DIR, "stubs")
    os.makedirs(name=STUBS_DIR, exist_ok=True)

    video_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    read_from_stubs = bool(int(sys.argv[3]))
    print("READ FROM STUBS: ", read_from_stubs)
    video_full_path = os.path.join(ASSETS_DIR, video_name)



    video_frames = read_video_as_frames(video_path=video_full_path)

    tracker = Tracker(MODEL_PATH, batch_size=batch_size)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stubs=True,
                                       stubs_path=STUBS_DIR+'/stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=read_from_stubs,
                                                                                stub_path=f'{STUBS_DIR}/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_frames,tracks)


    write_frames_as_video(output_frames, OUTPUT_VIDEO_PATH)



if __name__ == "__main__":
    main()