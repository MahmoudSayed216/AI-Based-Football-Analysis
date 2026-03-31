from ultralytics import YOLO
import supervision as sv


class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()


    def __detect_frames(self, frames):
        batch_size = 1
        detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            predictions = self.model.predict(source=batch, conf=0.1)
            detections.extend(predictions)
            break

        return detections
    

    def get_object_tracks(self, frames):
        detections = self.__detect_frames(frames=frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv_mapping = {v:k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            