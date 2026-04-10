from ultralytics import YOLO
import supervision as sv
import pickle
from utils import get_bbox_center, get_bbox_width
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


    def __detect_frames(self, frames):
        
        detections = []
        n_iters = len(frames)/self.batch_size
        curr = 1
        for i in range(0, len(frames), self.batch_size):
            curr+=1
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
    

    def draw_ellips(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)


        cv2.ellipse(frame, 
                    center=(x_center, y2), 
                    axes=(int(width), int(0.35*width)), 
                    angle=0.0, 
                    startAngle= -45, 
                    endAngle=235, 
                    color=color, 
                    thickness=2, 
                    lineType=cv2.LINE_4)
    
        RECTANGLE_WIDTH = 40
        RECTANGLE_HEIGHT = 20
        x1_rect = int(x_center - RECTANGLE_WIDTH//2)
        x2_rect = int(x_center + RECTANGLE_WIDTH//2)
        y1_rect = int(y2 - RECTANGLE_HEIGHT//2 + 15)
        y2_rect = int(y2 + RECTANGLE_HEIGHT//2 + 15)


        if track_id is not None:
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

            x1_text = x1_rect+11
            if track_id > 99:
                x1_text-=10

            cv2.putText(frame, f"{track_id}", (x1_text, y1_rect+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)

        

        return frame


    def draw_triangle(self, frame, bbox, color):
        DELTA = 10
        y = int(bbox[1])
        x_center, _ = get_bbox_center(bbox)
        triangle_pts = np.array([
            [x_center, y],
            [x_center-DELTA, y-DELTA-5],
            [x_center+DELTA, y-DELTA-5] 
        ])

        cv2.drawContours(frame, contours=[triangle_pts], contourIdx=0, color=color, thickness=cv2.FILLED)
        cv2.drawContours(frame, contours=[triangle_pts], contourIdx=0, color=(0, 0, 0), thickness=2)

        return frame


    def draw_annotations(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            players_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referees_dict = tracks["referees"][frame_num]


            for track_id, player in players_dict.items():
                frame = self.draw_ellips(frame, player["bbox"], (0, 0, 255),track_id)

            for track_id, referee in referees_dict.items():
                frame = self.draw_ellips(frame, referee["bbox"], (0, 255, 0))
            
            for track_id, ball in ball_dict.items():
                # frame = self.draw_ellips(frame, ball, (255, 255, 0),track_id)
                if ball:
                    frame = self.draw_triangle(frame, ball["bbox"], (255, 255, 255))

            output_frames.append(frame)


        return output_frames