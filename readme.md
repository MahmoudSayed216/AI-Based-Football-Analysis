Football Analysis Using Yolo and ML

After training almost all yolo version from 8-12 as well as 26 [the latest version of yolo] with all sizes except for the x size
Yolo26l had the best overall metrics for this task


## YOLOv6l Validation Metrics

| Class      | Images | Instances | Box (P) | R     | mAP50 | mAP50-95 |
|------------|--------|-----------|---------|-------|-------|----------|
| all        | 38     | 905       | 0.964   | 0.826 | 0.882 | 0.641    |
| ball       | 35     | 35        | 0.970   | 0.457 | 0.589 | 0.284    |
| goalkeeper | 27     | 27        | 0.926   | 0.925 | 0.966 | 0.760    |
| player     | 38     | 754       | 0.984   | 0.984 | 0.994 | 0.818    |
| referee    | 38     | 89        | 0.977   | 0.936 | 0.980 | 0.702    |

A couple of notes worth mentioning:

- Player detection is the strongest class overall, with near-perfect precision and recall.
- Ball detection is expectedly the weakest — it's a small, fast-moving object, so the lower recall (0.457) and mAP50-95 (0.284) are pretty typical for this kind of task.
- Inference speed was 21.0ms on T4 x2 per image, which is solid for a real-time use case | and ~250.0 ms per image on Intel(R) Core(TM) i7-1355U.
<br>
<br>
**How to Run:**

download a video for inference [samplevideo] and place it in the assets directory.

install requirements
```
pip install -r requirements.txt
```

if you want to retrain the yolo models on a new dataset, or train newer versions of yolo on the same dataset, you can go ahead with these 3 steps, otherwise, just use the weights provided on [download the .pt file from here {put google drive url here} and place it in a directory called "best_model" within the `BASE_DIR` ] and skip these steps.

__start of steps__

download yolo models
```python
python3 download_yolo_models.py
```

download the dataset
```python
python3 download_dataset.py
```

train the models
```python
python3 train_grid_and_save_best_model.py
```

use the trained modoel for inference

__end of steps__
 




```python
python3 main.py video_file_name
```
