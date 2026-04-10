import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        player_img = frame[y1:y2, x1:x2]

        # Use top half of bounding box (jersey area, avoid shorts/legs)
        top_half = player_img[:player_img.shape[0] // 2, :]

        kmeans = self.get_clustering_model(top_half)

        labels = kmeans.labels_
        clustered = labels.reshape(top_half.shape[:2])

        # The corner pixels are most likely background
        corners = [clustered[0, 0], clustered[0, -1], clustered[-1, 0], clustered[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - bg_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_data in player_detections.items():
            bbox = player_data['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        # Goalkeepers often look different — hardcode if needed later
        self.player_team_dict[player_id] = team_id
        return team_id