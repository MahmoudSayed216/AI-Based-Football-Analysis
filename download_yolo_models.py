import requests
import os
from dotenv import load_dotenv
import shutil

load_dotenv(".env")



BASE_DIR = os.getenv("BASE_DIR")
print(BASE_DIR)

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
# os.makedirs(WEIGHTS_DIR, exist_ok=True)



def download_weights(weights_url, save_path):
    response = requests.get(weights_url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    pass






models = {
    "Yolov08": ["https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
               "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
               "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt",
               "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l.pt"],
    
    "Yolov09": ["https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9t.pt",
               "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9s.pt",
               "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9m.pt",
               "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov9c.pt"],

    "Yolov10": ["https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10n.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10s.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10m.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10b.pt",],

    "Yolov11": ["https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt"],
    
    "Yolov12": ["https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12m.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12l.pt"],

    "Yolov26": ["https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt"]
}


for name, urls in models.items():
    print(f"name: {name}")
    for url in urls:
        print(url)
    print("---")




if os.path.exists(WEIGHTS_DIR):
    shutil.rmtree(WEIGHTS_DIR)


for name in models.keys():
    model_version_variants_path = os.path.join(WEIGHTS_DIR, name)
    os.makedirs(name=model_version_variants_path, exist_ok=True)
    urls = models[name]
    for i, url in enumerate(urls):
        model_variant_name = url[url.rfind("/")+1:]
        print(f"downloading {model_variant_name} ...")
        save_path = os.path.join(model_version_variants_path, f"{i}{model_variant_name}")
        download_weights(weights_url=url, save_path=save_path)
        print(f"{model_variant_name} downloaded to {save_path}.")

    

    

print("_________________")
print("Download complete")
print("_________________")
