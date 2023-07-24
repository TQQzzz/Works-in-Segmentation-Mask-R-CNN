import argparse
import glob
import os, json, cv2, random
import os.path as osp
import shutil
import xml.etree.ElementTree as ET
import wandb

import numpy as np
from tqdm import tqdm
import cv2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.config.config import CfgNode as CN
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import ResNet, build_resnet_backbone
from detectron2.engine import PeriodicWriter
from detectron2.engine import hooks

import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_score, recall_score
import argparse


outside_features = ['WINDOW','PC','BRICK','TIMBER','LIGHT-ROOF','RC2-SLAB','RC2-COLUMN']
inside_features = ['RC-SLAB','RC-JOIST','RC-COLUMN','TIMBER-COLUMN','TIMBER-JOIST']

def main(class_names_to_keep):
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


    ###########################################################################################
    # This part is to convert the labels-begin
    ###########################################################################################
    import os
    import json

    # Function to filter and keep only desired classes in the COCO dataset
    def filter_classes(input_dir, output_dir, class_names_to_keep):
        '''
        This function filters classes in a COCO formatted dataset. It takes in the
        original dataset, selects only the specified classes, and writes the 
        filtered dataset to an output directory.

        Args:
            input_dir (str): The directory where the original COCO formatted dataset is located.
            output_dir (str): The directory where the filtered COCO formatted dataset will be saved.
            class_names_to_keep (list): A list of class names that should be kept in the filtered dataset.
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for dataset_type in ["train", "val"]:
            input_file = os.path.join(input_dir, f"{dataset_type}.json")
            output_file = os.path.join(output_dir, f"{dataset_type}{'_'.join(class_names_to_keep)}.json")

            with open(input_file, "r") as f:
                data = json.load(f)

            new_data = {"images": data["images"], "annotations": [], "categories": []}

            # Filter out unwanted categories and only keep the specified classes
            category_ids_to_keep = []
            for category in data["categories"]:
                if category["name"] in class_names_to_keep:
                    category_ids_to_keep.append(category["id"])
                    new_data["categories"].append(category)

            # Keep only the annotations of the specified classes
            for annotation in data["annotations"]:
                if annotation["category_id"] in category_ids_to_keep:
                    new_data["annotations"].append(annotation)

            # Save the filtered annotations to a new file
            with open(output_file, "w") as f:
                json.dump(new_data, f, indent=2)

            print(f"Processed {dataset_type} dataset:")
            print(f"  Categories: {len(new_data['categories'])}")
            print(f"  Annotations: {len(new_data['annotations'])}")
        print('Filter the labels and keep only the specified classes-----Succeed')
        print('##############################################')

    #Register dataset
    from detectron2.data.datasets import register_coco_instances


    # Example usage
    input_dir_inside = "/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/annotations"
    output_dir_inside = "/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/annotations/modified/"

    input_dir_outside = "/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/annotations"
    output_dir_outside = "/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/annotations/modified/"

    #the inside and outside should have different input and output folder
    if class_names_to_keep[0] in inside_features:

        filter_classes(input_dir_inside, output_dir_inside, class_names_to_keep)
        register_coco_instances("my_dataset_train", {}, f"/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/annotations/modified/train{'_'.join(class_names_to_keep)}.json", "/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/train")
        register_coco_instances("my_dataset_val", {}, f"/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/annotations/modified/val{'_'.join(class_names_to_keep)}.json", "/scratch/tz2518/Segmentation_MRCNN/!data_single_in/coco/val")

    else:
        filter_classes(input_dir_outside, output_dir_outside, class_names_to_keep)
        register_coco_instances("my_dataset_train", {}, f"/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/annotations/modified/train{'_'.join(class_names_to_keep)}.json", "/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/train")
        register_coco_instances("my_dataset_val", {}, f"/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/annotations/modified/val{'_'.join(class_names_to_keep)}.json", "/scratch/tz2518/Segmentation_MRCNN/!data_single_out/coco/val")
  
    #class_names_to_keep = ["Window"]  # Add or remove class names as needed
   

    ###########################################################################################
    # This part is to convert the labels-end
    ###########################################################################################

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


    ###########################################################################################
    # This part is to train the model
    ###########################################################################################

   
    # Initialize wandb
    #wandb.init(project="Single", sync_tensorboard=True,name=class_names_to_keep)
    wandb.init(project="Single_new", sync_tensorboard=True, name='_'.join(class_names_to_keep))

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
    cfg.OUTPUT_DIR = f"/scratch/tz2518/Segmentation_MRCNN/{'_'.join(class_names_to_keep)}"

    #cfg setting parameters-----End
    #--- For this part, both train.py and test.py should have the same parameters.
    ###############################################################

    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader


    class MyTrainer(DefaultTrainer):
        # This is a custom trainer that overrides some of the default methods in DefaultTrainer.
        # overrides the methods can help wandb record the loss when the model is running
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            # This method is used to create an evaluator for the validation phase.
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

        def build_hooks(self):
            # This method is used to build the list of hooks that will be used in training.
            # It adds WandbLoggerHook to the default hooks.
            hooks = super().build_hooks()
            hooks.insert(-1, PeriodicWriter([WandbLoggerHook(self)], period=self.cfg.TEST.EVAL_PERIOD))
            return hooks



    from detectron2.utils.events import EventWriter
    class WandbLoggerHook(EventWriter):
        # This is a custom EventWriter that logs losses to Weights and Biases (wandb).
        def __init__(self, trainer):
            self.trainer = trainer

        def write(self):
            # This method logs the losses to Weights and Biases at each step.
            storage = self.trainer.storage
            metrics_dict = storage.latest()
            loss_dict = {k: v for k, v in metrics_dict.items() if "loss" in k}
            wandb.log(loss_dict, step=storage.iter)



    from detectron2.data import DatasetMapper, build_detection_train_loader
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.transforms import Augmentation
    from detectron2.data import transforms as T

    # The following block defines the data augmentation transformations 
    # to be applied on the training dataset.
    augmentation = [
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # Horizontal flip
        T.RandomCrop(crop_type="relative", crop_size=(0.9, 0.9)),  # Random crop, crop size is 0.9 times the original size
        T.RandomRotation(angle=[-10, 10]),  # Random rotation angle range [-10, 10]
        T.RandomBrightness(0.8, 1.2),  # Random brightness adjustment range [0.8, 1.2]
        T.RandomContrast(0.8, 1.2),  # Random contrast adjustment range [0.8, 1.2]
    ]

    # Define a new function that will create the data loader with the augmentation
    def build_detection_train_loader_with_transform(cfg, mapper=None):
        # This function creates a data loader with the specified data augmentation transformations.
        if mapper is None:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentation)
        return build_detection_train_loader(cfg, mapper=mapper)

    # Initialize the custom trainer and set the custom data loader.
    trainer = MyTrainer(cfg)
    trainer.build_train_loader = build_detection_train_loader_with_transform
    trainer.train()

if __name__ == '__main__':
    # The main part of the script that takes the list of classes 
    # from the command-line arguments and starts the training.
    parser = argparse.ArgumentParser(description="Train a detectron2 model with specified classes.")
    parser.add_argument("classes", nargs='+', help="List of classes to keep. E.g., Window Door")

    args = parser.parse_args()

    main(args.classes)