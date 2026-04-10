import sys 
from utils import get_bbox_center, calculate_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_bbox_center(ball_bbox)

        miniumum_distance = float('inf')
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = calculate_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = calculate_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player