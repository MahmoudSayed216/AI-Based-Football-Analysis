from ultralytics import YOLO
import supervision as sv
import pickle
from utils import get_bbox_center, get_bbox_width, get_foot_position
import cv2
import numpy as np
import pandas as pd
# class TID_BBOX_PAIR:
#     def __init__(self, track_id: int, bbox: list[float]):
#         self.track_id = track_id
#         self.bbox = bbox


class Tracker:

    def __init__(self, model_path, batch_size = 20):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()
        self.batch_size = batch_size

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_bbox_center(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position


    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def __detect_frames(self, frames):
        
        detections = []
        n_iters = len(frames)/self.batch_size
        curr = 1
        for i in range(0, len(frames), self.batch_size):
            curr+=1
            print("1")
            batch = frames[i:i+self.batch_size]
            predictions = self.model.predict(source=batch, conf=0.1, verbose=False)
            detections.extend(predictions)

        return detections
    
    def interpolate_ball_positions(self,ball_positions):
            ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
            df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

            # Interpolate missing values
            df_ball_positions = df_ball_positions.interpolate()
            df_ball_positions = df_ball_positions.bfill()

            ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

            return ball_positions
    


    def get_object_tracks(self, frames, read_from_stubs=False, stubs_path=None):
        
        if read_from_stubs:
            if stubs_path is None:
                raise ValueError("stubs path is None")
            
            with open(stubs_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks

        detections = self.__detect_frames(frames=frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv_mapping = {v:k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            for idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[idx] = cls_names_inv_mapping["player"]
                    detection_supervision.data['class_name'][idx] = "player"


            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)


            # print(detections_with_tracks)
            # print("###########################################")
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for detection in detections_with_tracks:
                bbox = detection[0].tolist()
                           #data idx = 5
                class_name = detection[5]['class_name']



                if class_name == "referee":
                    track_id = detection[4]
                    tracks["referees"][frame_num][track_id] = {'bbox': bbox}
                elif class_name == "player":
                    track_id = detection[4]
                    tracks["players"][frame_num][track_id] = {'bbox': bbox}

                elif class_name == "ball":
                    tracks["ball"][frame_num][1] = {'bbox': bbox}

        
        if stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(tracks, f)


        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_bbox_center(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,125))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames