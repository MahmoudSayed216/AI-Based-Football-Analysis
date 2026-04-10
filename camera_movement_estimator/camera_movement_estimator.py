import cv2
import numpy as np
import pickle
import os


class CameraMovementEstimator:
    def __init__(self, first_frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Mask out the center (where players move) — only track edges for camera motion
        mask_features = np.zeros_like(first_gray)
        mask_features[:, :20] = 1        # left strip
        mask_features[:, -20:] = 1       # right strip

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(self, frames, read_from_stubs=False, stubs_path=None):
        if read_from_stubs and stubs_path and os.path.exists(stubs_path):
            with open(stubs_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for new, old, st in zip(new_features, old_features, status):
                if st[0] == 1:
                    new_pt = new.ravel()
                    old_pt = old.ravel()
                    distance = np.linalg.norm(new_pt - old_pt)
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x = new_pt[0] - old_pt[0]
                        camera_movement_y = new_pt[1] - old_pt[1]

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stubs_path:
            with open(stubs_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position', None)
                    if position is None:
                        continue
                    cam_x, cam_y = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - cam_x, position[1] - cam_y)
                    tracks[obj][frame_num][track_id]['position_adjusted'] = position_adjusted

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()

            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x, y = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames