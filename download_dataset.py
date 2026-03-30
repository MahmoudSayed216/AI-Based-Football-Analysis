import os
from dotenv import load_dotenv
from roboflow import Roboflow
import shutil

load_dotenv(".env")


API_KEY = os.getenv("API_KEY")
ROBOFLOW_WORKSPACE  = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_WORKSPACE_PROJECT = os.getenv("ROBOFLOW_WORKSPACE_PROJECT")
BASE_DIR = os.getenv("BASE_DIR")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)



rf = Roboflow(api_key=API_KEY)
project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_WORKSPACE_PROJECT)
version = project.version(1)
dataset = version.download("yolov11", location=DATASET_DIR)