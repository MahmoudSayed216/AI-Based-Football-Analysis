from ultralytics import YOLO
import supervision as sv
import pickle
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
            print(f"iteration {curr}/{n_iters}")
            curr+=1
            batch = frames[i:i+self.batch_size]
            predictions = self.model.predict(source=batch, conf=0.1, verbose=False)
            detections.extend(predictions)

        return detections
    

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
                    tracks["referees"][frame_num][track_id] = bbox
                elif class_name == "player":
                    track_id = detection[4]
                    tracks["players"][frame_num][track_id] = bbox

                elif class_name == "ball":
                    tracks["ball"][frame_num][1] = bbox

        
        if stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(tracks, f)


        return tracks