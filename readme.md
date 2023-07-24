- `Segmentation_MRCNN/`
    - `dataProcess_evaluation.py`  (useless)
    - `evaluation_single_full_data.py` (useless)
    - `dataProcess_single_out.py`
    - `dataProcess_single_in.py`
    - `train_single_MRCNN.py`
    - `evaluation_single_in_out.py`


--- 
--- 

### **Segmentation-Mask-RCNN**

- **Data Process**

  ```
  python dataProcess_single_out.py
  python dataProcess_single_in.py
  ```

  Entering the commands mentioned above in the terminal will create folders '!data_single_in' and '!data_single_out', which can be used for training data.

- **Training Models and Get the Output Folder**

  ```
  python train.py WINDOW
  ```

  Entering the command python train.py {Class_name} to start training the model and then a folder will be created and the path is Segmentation_MRCNN/WINDOW.  Similarly commands can be used for other classes.

- **Evaluate the model**

  ```
  python evaluation_single_in_out.py
  ```

  Entering this command to evaluate both inside and outside models and get an excel.

