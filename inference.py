from ultralytics import YOLO
from dotenv import load_dotenv
import os
load_dotenv('.env')

BASE_DIR = os.getenv('BASE_DIR')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')



model = YOLO(model="./best_model/weights/best.pt")

model.predict(source="./assets/08fd33_4.mp4", save=True)