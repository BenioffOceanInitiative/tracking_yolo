# YoloV5 with SORT implementation And DeepSORT implementation

### Based on Ultralytics [YoloV5](https://github.com/ultralytics/yolov5) and [SORT](https://github.com/abewley/sort)

## Installation
 ```
    git clone https://github.com/BenioffOceanInitiative/tracking_yolo.git
 ```

## Usage
 Use conda to create a virtual environment

 ```
    conda create -n yolo_sort python=3.8
    conda activate yolo_sort
    pip install -r requirements.txt
 ```

## Run DeepSORT tracking

```
python DeepSort.py --weights <path to .pt file> --source<path to video file or specify camera> --show-vid (to show video) --save-vid(to save video)
```

 ## Run tracking with totals output
 ```
 python track_with_totals_output.py
 ```

