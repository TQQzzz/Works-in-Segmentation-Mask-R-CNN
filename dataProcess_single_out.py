import os
import shutil
import json
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
import base64
import skimage.draw
import numpy as np
from pycocotools import mask as cocomask

# This is the features going to be saved, for the inside ones will be deleted
class_map = {
    "WINDOW": 0,
    "PC": 1,
    "BRICK": 2,
    "TIMBER":3,
    "LIGHT-ROOF": 4,
    "RC2-SLAB":5,
    "RC2-COLUMN":6
}




def divide_files_into_folders(main_folder,output_folder_images,output_folder_jsons):
    '''
    Divides files in the main folder into separate folders for images and JSON files.

    Args:
        main_folder (str): The path to the main folder.
        output_folder_images (str): The path to the folder for images.
        output_folder_jsons (str): The path to the folder for JSON files.
    '''

    # Iterate through the subfolders in the main folder
    # Iterate through the subfolders in the main folder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        # Check if the path is a folder and not an output folder
        if os.path.isdir(subfolder_path) and subfolder != output_folder_images and subfolder != output_folder_jsons:
            
            # Iterate through the files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.startswith("._"):
                    # Delete the file
                    file_path = os.path.join(subfolder_path, file)
                    os.remove(file_path)
                    continue

                file_path = os.path.join(subfolder_path, file)

                # Check if the file is an image (assuming .jpg or .png format)
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Check if the filename contains the word "Building"
                    if "building" in file.lower():
                    #if "diaphragm" in file.lower():
                    #if 1:
                        # Copy the image to the images folder
                        shutil.copy(file_path, output_folder_images)

                # Check if the file is a .json file
                elif file.lower().endswith(".json"):
                    # Check if the filename contains the word "Building"
                    if "building" in file.lower():
                    #if "diaphragm" in file.lower():
                    #if 1:
                        # Copy the .json file to the json_files folder
                        shutil.copy(file_path, output_folder_jsons)

    print("Files have been successfully divided into images and json_files folders within the data folder.")

def change_label_name(label):
    #this part is to transform the old names to new ones.
    label_dict = {
        "Window": "Window",
        "Precast-RC-slabs": "RC-Slab",
        #"RC-solid-slab": "RC-Slab"
        "RC-solid-slab": "RC2-Slab",# special for the outside 
                                    #some outside slabs was wrong labeled as RC-solid-slab
                                    #because the function divide_files_into_folders run first
                                    #So it will not transfer the inside images
        "RC-Joist": "RC-Joist",
        "PC1": "PC",
        "PC2": "PC",
        #"RC-Column": "RC-Column",
        "RC-Column": "RC2-Column",# special for the outside, 
                                #some outside columns was wrong labeled as RC-Column
        "Slab": "RC2-Slab",
        "UCM/URM7": "Brick",
        "Timber-Frame": "Timber",
        "Timber-Column": "Timber-Column",
        "Timber-Joist": "Timber-Joist",
        "Light-roof": "Light-roof",
        "UCM/URM4": "Brick",
        "RM1": "Masonry",
        "Adobe": "Brick",
        'RC2-Column':'RC2-Column'#outside
    }
   

    return label_dict.get(label, label)

def clean_label(label):
    '''
    Cleans a label by removing trailing 's' and converting it to uppercase.

    Args:
        label (str): The label to clean.

    Returns:
        str: The cleaned label.
    '''
    # Remove trailing 's'
    if label.endswith('s'):
        label = label[:-1]

    # Convert to uppercase
    label = label.upper()

    return label

def clean_labels(data_folder):
    '''
    Cleans and transfer the labels in JSON files within the specified data folder. 

    Args:
        data_folder (str): The path to the data folder.
    '''
    for file in os.listdir(data_folder):
        if file.lower().endswith(".json"):
            # Load the .json file
            with open(os.path.join(data_folder, file), 'r') as json_file:
                data = json.load(json_file)
                new_shapes = []

                for obj in data['shapes']:
                    print(obj['label'])
                    #Change the label name 
                    new_label = change_label_name(obj['label'])  

                    print(new_label)

                    # clean the label
                    new_label = clean_label(new_label)
                    print(new_label)

                    if new_label in class_map:
                        obj['label'] = new_label
                        print(new_label)
                        new_shapes.append(obj)
                    
                    print("----------------------------")

                # Replace the old shapes with the new ones
                data['shapes'] = new_shapes

            # Save the cleaned json file
            with open(os.path.join(data_folder, file), 'w') as json_file:
                json.dump(data, json_file)
    print('clean successfully')

def convert_folder_to_coco(image_folder, json_folder, coco_folder):
    '''
    Converts the images and corresponding annotations in the input folders into the COCO format.

    Args:
        image_folder (str): The path to the folder containing the images.
        json_folder (str): The path to the folder containing the JSON annotations.
        coco_folder (str): The path to the output COCO folder.
    '''
    #create the path for the folders
    coco_train = os.path.join(coco_folder, 'train')
    coco_val = os.path.join(coco_folder, 'val')
    coco_annotations = os.path.join(coco_folder, 'annotations')

    # Create the necessary folders if they do not exist
    os.makedirs(coco_train, exist_ok=True)
    os.makedirs(coco_val, exist_ok=True)
    os.makedirs(coco_annotations, exist_ok=True)

    all_files = os.listdir(json_folder)
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)  # 80% for training, 20% for validation

    def handle_files(files, dataset_type):
        '''
        Convert the json files format to coco

        Args:
            files (list): List of file names.
            dataset_type (str): The type of the dataset ('train' or 'val').
        '''
        output_folder_images = os.path.join(coco_folder, dataset_type)
        output_folder_labels = os.path.join(coco_annotations, dataset_type + '.json')

        os.makedirs(output_folder_images, exist_ok=True)

        # Similar to your previous coco_data generation
        coco_data = {
            'info': {},
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        }

        category_id_map = {}
        category_id = 1
        annotation_id = 1

        for filename in files:
            if filename.endswith('.json'):
                json_path = os.path.join(json_folder, filename)
                image_path = os.path.join(image_folder, f"{Path(filename).stem}.jpg")

                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, f"{Path(filename).stem}.jpeg")
                    if not os.path.exists(image_path):
                        #print(f"Skipping file: {image_path} not found")
                        continue



                try:
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)

                    img_id = len(coco_data['images']) + 1
                    img = Image.open(image_path)
                    # width, height = img.size
                    width = int(json_data['imageWidth'])
                    height = int(json_data['imageHeight'])

                    if img._getexif() is not None:
                        exif_data = img._getexif()
                        orientation = exif_data.get(274, 1)
                        if orientation in [5, 6, 7, 8]:
                            width, height = height, width

                    coco_data['images'].append({
                        'id': img_id,
                        'width': width,
                        'height': height,
                        'file_name': Path(image_path).name,
                    })

                    for shape in json_data['shapes']:
                        label = shape['label']

                        if label not in category_id_map:
                            category_id_map[label] = category_id
                            category_id += 1

                            coco_data['categories'].append({
                                'id': category_id_map[label],
                                'name': label,
                                #'supercategory': 'building',
                            })

                        flat_points = [coord for point in shape['points'] for coord in point]

                        binary_mask = np.zeros((height, width), dtype=np.uint8)
                        y_coords = np.clip([point[1] for point in shape['points']], 0, height - 1)
                        x_coords = np.clip([point[0] for point in shape['points']], 0, width - 1)
                        rr, cc = skimage.draw.polygon(y_coords, x_coords)
                        binary_mask[rr, cc] = 1

                        rle = cocomask.encode(np.asfortranarray(binary_mask))
                        segmentation = [flat_points]

                        coco_data['annotations'].append({
                            'id': annotation_id,
                            'image_id': img_id,
                            'category_id': category_id_map[label],
                            'segmentation': segmentation,
                            'area': float(cocomask.area(rle)),
                            'bbox': cocomask.toBbox(rle).tolist(),
                            'iscrowd': 0,
                        })

                        annotation_id += 1
                    
                    # Copy the image to the corresponding output folder
                    shutil.move(image_path, output_folder_images)


                except Exception as e:
                    #print(f"Skipping file: {json_path} due to an error: {str(e)}")
                    continue
            
        with open(output_folder_labels, 'w') as f:
            json.dump(coco_data, f)

    handle_files(train_files, 'train')
    handle_files(val_files, 'val')


#!!! main_folder is the downloaded data folder, the output_folder is the path you want to save
main_folder = '/scratch/tz2518/data_6.14'
output_folder = '/scratch/tz2518/Segmentation_MRCNN/!data_single_out'  # Specify the output folder
os.makedirs(output_folder, exist_ok=True)


#make the folders in the output_folder
image_folder = os.path.join(output_folder, 'image')
os.makedirs(image_folder, exist_ok=True)
json_folder = os.path.join(output_folder, 'json')
os.makedirs(json_folder, exist_ok=True)
coco_folder = os.path.join(output_folder, 'coco')
os.makedirs(output_folder, exist_ok=True)

#divide the data to images and json
divide_files_into_folders(main_folder, image_folder, json_folder)
#clean the json files
clean_labels(json_folder)
#convert the json files to coco format
convert_folder_to_coco(image_folder, json_folder, coco_folder)