import cv2
import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config.config import CfgNode as CN

class_map = {
    "WINDOW": 0,
    "PC": 1,
    "BRICK": 2,
    "TIMBER":3,
    "LIGHT-ROOF": 4,
    "RC2-SLAB":5,
    "RC2-COLUMN":6,
    "RC-SLAB": 7,
    "RC-JOIST": 8,
    "RC-COLUMN": 9,
    "TIMBER-COLUMN": 10,
    "TIMBER-JOIST": 11
}


def evaluation_detection(class_names_to_keep,Threshold):

    ###############################################################
    #cfg setting parameters-----Begin
    cfg = get_cfg()
    cfg.NUM_GPUS = 1
    # Dataset Configuration
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    # Input Configuration
    cfg.INPUT.MIN_SIZE_TRAIN = (1024, 960)
    cfg.INPUT.MAX_SIZE_TEST = 1700
    cfg.INPUT.MAX_SIZE_TRAIN = 1000

    # Model Configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.MASK_ON = True  # Ensure mask prediction is enabled

    # Anchor Generator Configuration
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    # Normalization Configuration
    cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    # Solver Configuration
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.STEPS = (12000, 18000)
    cfg.SOLVER.GAMMA = 0.2

    # Evaluation Configuration
    cfg.TEST.EVAL_PERIOD = 2000  # Add this line to set the evaluation period

    # ResNeXt Configuration
    cfg.MODEL.RESNEXT = CN()
    cfg.MODEL.RESNEXT.DEPTH = 152
    cfg.MODEL.RESNEXT.NUM_GROUPS = 64
    cfg.MODEL.RESNEXT.WIDTH_PER_GROUP = 32

    # Output Configuration
    #cfg.OUTPUT_DIR = f"/scratch/tz2518/Segmentation_MRCNN/{class_names_to_keep}"
    cfg.OUTPUT_DIR = f"/scratch/tz2518/Segmentation_MRCNN/{(class_names_to_keep)}"


    #cfg setting parameters-----End
    ######--- For this part, both train.py and test.py should have the same parameters.

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Threshold
    cfg.MODEL.RETINANET = False


    predictor = DefaultPredictor(cfg)
    
    #metadata = MetadataCatalog.get("my_dataset_train")
    

    
    image_folder= '/scratch/tz2518/Segmentation_MRCNN/data_evaluation/coco/train'
    val_json_path = '/scratch/tz2518/Segmentation_MRCNN/data_evaluation/coco/annotations/train.json'
    
    # 加载验证集的标签
    with open(val_json_path) as f:
        val_labels = json.load(f)

    # 创建一个字典，将 image_id 映射到 file_name
    id_to_filename = {img['id']: img['file_name'] for img in val_labels['images']}

    # initialize the counter
    TP = 0
    FP = 0
    TN = 0
    FN = 0


    predictions = []
    true_labels = []

    for image_dict in val_labels['images']:
        image_file_name = image_dict['file_name']
        image_path = os.path.join(image_folder, image_file_name)

        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            continue

        try:
            im = cv2.imread(image_path)
            if im is None:
                raise IOError
        except IOError:
            print(f"Failed to load image: {image_path}")
            continue

        outputs = predictor(im)

        # 检查模型是否预测出了掩模
        prediction = outputs["instances"].pred_masks.shape[0] > 0
        predictions.append(prediction)

        # 获取真实标签
        true_label = 0
        image_id = image_dict['id']

        for ann in val_labels['annotations']:
            if ann['image_id'] == image_id:
                true_label = 1
                break
        true_labels.append(true_label)

        # update the counter
        if prediction and true_label:
            TP += 1
        elif prediction and not true_label:
            FP += 1
        elif not prediction and true_label:
            FN += 1
        elif not prediction and not true_label:
            TN += 1

    # calculate the scores
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)

    print('For the class:', class_names_to_keep)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print('TP',TP,'FP',FP,'FN',FN,'TN',TN)
    print('---------------------------------------------')

    return TP, FP, FN, TN, accuracy, precision, recall





# 初始化一个空的 DataFrame 来存储结果
results = pd.DataFrame(columns=['Feature', 'Threshold', 'TP', 'FP', 'FN', 'TN', 'Accuracy', 'Precision', 'Recall'])

features = ["WINDOW", "RC-SLAB", "RC-JOIST", "PC", "RC-COLUMN", "RC2-SLAB", "BRICK", 
            "TIMBER", "TIMBER-COLUMN", "TIMBER-JOIST", "LIGHT-ROOF", "RC2-COLUMN"]


for feature in features:
    for i in range(90, 100):
        start_time = time.time()
        threshold = i / 100.0
        print(threshold)
        TP, FP, FN, TN, accuracy, precision, recall = evaluation_detection(feature, threshold)
        elapsed_time = time.time() - start_time
        print('time-consuming:',elapsed_time)
        results.loc[len(results)] = [feature, threshold, TP, FP, FN, TN, accuracy, precision, recall]