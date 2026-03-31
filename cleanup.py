import os
import shutil
import sys
from dotenv import load_dotenv




def delete_unnecessary_files(base_dir, delete_dataset= True):
    
    DATASET_DIR = os.path.join(base_dir, 'dataset')
    WEIGHTS_DIR = os.path.join(base_dir, 'weights')
    RUNS_DIR = os.path.join(base_dir, 'runs')

    if os.path.exists(DATASET_DIR) and delete_dataset == True:
        shutil.rmtree(DATASET_DIR)

    if os.path.exists(WEIGHTS_DIR):
        shutil.rmtree(WEIGHTS_DIR)

    if os.path.exists(RUNS_DIR):
        shutil.rmtree(RUNS_DIR)




def copy_yolo_pics_to_assets_dir(base_dir):
    ASSETS_DIR = os.path.join(base_dir, "assets")
    OUTPUT_DIR = os.path.join(base_dir, "best_model")
    os.makedirs(ASSETS_DIR, exist_ok=True)

    output_dir_content = os.listdir(OUTPUT_DIR)
    output_dir_content.remove('weights')
    output_dir_content.remove('results.csv')
    output_dir_content.remove('args.yaml')

    for content in output_dir_content:
        content_full_path = os.path.join(OUTPUT_DIR, content)
        shutil.copy2(content_full_path, ASSETS_DIR)



def main():

    delete_ds = bool(sys.argv[1])


    load_dotenv('.env')
    BASE_DIR = os.getenv('BASE_DIR')


    delete_unnecessary_files(BASE_DIR, delete_dataset=delete_ds)
    copy_yolo_pics_to_assets_dir(BASE_DIR)



if __name__ == "__main__":
    main()
