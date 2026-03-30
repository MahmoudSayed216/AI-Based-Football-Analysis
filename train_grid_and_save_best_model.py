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
DATASET_DIR = os.path.join(BASE_DIR, "dataset","data.yaml")
RUN_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(name=OUTPUT_DIR, exist_ok=True)
os.makedirs(name=RUN_DIR, exist_ok=True)

current_best_map = -100




def train_model(model_path, model_name):
    global current_best_map
    print(f"Training {model_name}")
    model = YOLO(model=model_path, task='detect', verbose=False)
    results = model.train(data=DATASET_DIR,
                epochs=2, 
                imgsz=640, 
                batch=16, 
                # save=False,
                # plots=False,
                save_period=-1,
                exist_ok=True,
                # val=False,
                project=RUN_DIR,
                name=model_name,
                amp=False,
                cos_lr = True,
                device=[0, 1],  # use both GPUs
)


    val_map = results.box.map
    print(f"map@50-95: {val_map}")
    # if os.path.exists(RUN_DIR):
        # shutil.rmtree(RUN_DIR)
    RUN_SRC_DIR = os.path.join(RUN_DIR, f"{model_name}")
    if val_map > current_best_map:
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(name=OUTPUT_DIR, exist_ok=True)
        print("Model updated")
        current_best_map = val_map
        shutil.copytree(RUN_SRC_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    shutil.rmtree(RUN_SRC_DIR)
    

    print("################################")









model_versions = sorted(os.listdir(WEIGHTS_DIR))

for model_version in model_versions:
    model_version_dir = os.path.join(WEIGHTS_DIR, model_version)
    model_sizes = sorted(os.listdir(model_version_dir))
    
    for model_size in model_sizes:
        model_path = os.path.join(model_version_dir, model_size)

        train_model(model_path, model_size[1:])

