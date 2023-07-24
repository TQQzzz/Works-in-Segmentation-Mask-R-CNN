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


outside_features = ['WINDOW','PC','BRICK','TIMBER','LIGHT-ROOF','RC2-SLAB','RC2-COLUMN']
inside_features = ['RC-SLAB','RC-JOIST','RC-COLUMN','TIMBER-COLUMN','TIMBER-JOIST']

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
    #--- For this part, both train.py and test.py should have the same parameters.
    ###############################################################

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Threshold
    cfg.MODEL.RETINANET = False

    #use the default predictor to predic
    predictor = DefaultPredictor(cfg)
    
    

    if class_names_to_keep in inside_features:
        image_folder = '/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/train'
        val_json_path = f"/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/annotations/modified/train{(class_names_to_keep)}.json"
    if class_names_to_keep in outside_features:
        image_folder = '/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/train'
        val_json_path = f"/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/annotations/modified/train{(class_names_to_keep)}.json"

    # Loading tags for validation sets
    with open(val_json_path) as f:
        val_labels = json.load(f)

    # Create a dictionary that maps image_id to file_name
    id_to_filename = {img['id']: img['file_name'] for img in val_labels['images']}

    # initialize the counter
    TP = 0
    FP = 0
    TN = 0
    FN = 0


    predictions = []
    true_labels = []

    for image_dict in val_labels['images']:# Loop over images in the validation set       
        image_file_name = image_dict['file_name']# Get the filename of the image
        image_path = os.path.join(image_folder, image_file_name)# Construct the full path of the image

        # Check if the image exists, if it doesn't, skip this image
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            continue

        # Attempt to read the image, if the image cannot be loaded, skip it
        try:
            im = cv2.imread(image_path)
            if im is None:
                raise IOError
        except IOError:
            print(f"Failed to load image: {image_path}")
            continue

        outputs = predictor(im)

        # Get the true label of the image
        prediction = outputs["instances"].pred_masks.shape[0] > 0
        predictions.append(prediction)

        # get labels
        true_label = 0
        image_id = image_dict['id']

        for ann in val_labels['annotations']:
            if ann['image_id'] == image_id: # If the image ID of the annotation matches the current image ID
                true_label = 1# Set the true label as 1
                break
        true_labels.append(true_label)# Add the true label to the list of labels


        # Update the counts in the confusion matrix
        if prediction and true_label:  # TP: Both the true and the predicted labels are positive
            TP += 1
        elif prediction and not true_label:  # FP: The true label is negative but the predicted label is positive
            FP += 1
        elif not prediction and true_label:  # FN: The true label is positive but the predicted label is negative
            FN += 1
        elif not prediction and not true_label:  # TN: Both the true and the predicted labels are negative
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




results = pd.DataFrame(columns=['Feature', 'Threshold', 'TP', 'FP', 'FN', 'TN', 'Accuracy', 'Precision', 'Recall'])

features = ["WINDOW", "RC-SLAB", "RC-JOIST", "PC", "RC-COLUMN", "RC2-SLAB", "BRICK", 
            "TIMBER", "TIMBER-COLUMN", "TIMBER-JOIST", "LIGHT-ROOF", "RC2-COLUMN"]

#Iterate through each feature and threshold(0.90, 0.91, 0.92, .....0.99)
for feature in features:
    for i in range(90, 100):
        start_time = time.time()
        threshold = i / 100.0
        print(threshold)
        TP, FP, FN, TN, accuracy, precision, recall = evaluation_detection(feature, threshold)
        elapsed_time = time.time() - start_time
        print('time-consuming:',elapsed_time)
        results.loc[len(results)] = [feature, threshold, TP, FP, FN, TN, accuracy, precision, recall]

# save as Excel
results.to_excel("evaluation_results.xlsx", index=False)