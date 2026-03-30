import os
import shutil
from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from suppress_output import SuppressOutput


load_dotenv(".env")


SETTINGS['mlflow'] = False
BASE_DIR = os.getenv("BASE_DIR")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "best_model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "data.yaml")
os.makedirs(name=OUTPUT_DIR, exist_ok=True)

current_best_map = -100




def train_model(model_path, model_name):
    print(f"Training {model_name}")
    with SuppressOutput():
        model = YOLO(model=model_path, task='detect', verbose=False)
        model.train(data=DATASET_DIR,
                    epochs=2, 
                    imgsz=640, 
                    batch=16, 
                    verbose=False, 
                    save=False,
                    plots=False,
                    save_period=-1,
                    exist_ok=True,
                    val=False)


        val_map = model.metrics.box.map
        print(f"map@50-95: {val_map}")
        if val_map > current_best_map:
            current_best_map = val_map
            model.val(data=DATASET_DIR, plots=True)
            # shutil.copytree(src, OUTPUT_DIR, dirs_exist_ok=True)
            




    print("################################")






model_versions = sorted(os.listdir(WEIGHTS_DIR))

for model_version in model_versions:
    model_version_dir = os.path.join(WEIGHTS_DIR, model_version)
    model_sizes = sorted(os.listdir(model_version_dir))
    
    for model_size in model_sizes:
        model_path = os.path.join(model_version_dir, model_size)

        train_model(model_path, model_size)

