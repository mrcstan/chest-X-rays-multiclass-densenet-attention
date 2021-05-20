# Multi-Label Classification and Visual Highlight of Chest X-ray Images using Neural Networks with Attention Mechanism and Grad-CAM

Marcus Hwai Yik Tan, Xiaohan Tian, Wing Chan, Joshua Ceaser  
University of Illinois, Urbana-Champaign

## Quick Startn
- Set ALL_IMAGE_DIR to the folder containing the X-ray images
- Set BASE_PATH_LABELS to the folder containing the lists of training, validation and test image file names
- Run either of the following notebooks: 
  __t01-multilabel-main-test.ipynb__, 
  __t01-multilabel-non_image_features-main-test.ipynb__, 
  __t01-multilabel-main-val.ipynb__, 
  __t01-multilabel-non_image_features-main-val.ipynb__
  

## Guide

### For multi-label classification:
- __p02-dataset-selection-multilabel.ipynb__: This notebook can be skipped since the files containing the lists of selected images for the final report are already included in the "labels" folder. The files are train_val_A.csv, train_A_x.csv (x=1,2,3), val_A_x.csv (x=1,2,3) and test_A.csv. This notebook selects a subset of images for training, validation and test lists. Multiple training/validation splits are generated. The default folder is "labels", where Data_Entry_2017_v2020.csv is also located.
- __t01-multilabel-main-val.ipynb__: train and evaluate model on multiple training, validation splits
- __t01-multilabel-main-test.ipynb__: train model on training+validation dataset and evaluate model on a test dataset

### For multi-label classification with non-image features:
- __p02-dataset-add_non_image_features.ipynb__: Append non-image features to existing training, validation and test lists generated by p02-dataset-selection-multilabel.ipynb
- __t01-multilabel-non_image_features-main-val.ipynb__: same function as t01-multilabel-main-val.ipynb but with non-image features as additional inputs
- __t01-multilabel-non_image_features-main-test.ipynb__: same function as t01-multilabel-main-test.ipynb but with non-image features as additional inputs

### For statistics and evaluation:
- __t01-multilabel-test.py__: load a saved model and evaluate on a test dataset.
- __p02-dataset-stats.ipynb__: analysis chest x-ray dataset and draw statistic charts.
- __pp01-postprocess-performance.ipynb__: postprocess the performance stats in the performance directory.

### For heatmap generation using Grad-CAM:
- __t03-multilabel-heatmap-densenet121-v2.ipynb__: Load a saved model and draw heatmap image from given input image. Please note the `MODEL_NAME` can only be `densenet121`. A DenseNet-121 model trained on the images in the train_val_A.csv list for 8 epochs is provided in the models folder
- __t03-multilabel-heatmap-densenet121attA-v2.ipynb__: Load a saved model and draw heatmap image from given input image. Please note the `MODEL_NAME` can only be `densenet121attA`. A DenseNet-121-attA model trained on the images in the train_val_A.csv list for 8 epochs is provided in the models folder

### Python files containing standard and customized models:
- The following modules are built on the standard DenseNet model from https://github.com/pytorch/vision/blob/release/0.9/torchvision/models/densenet.py
    - __densenet_models.py__: standard DenseNet model and the additional functions to generate the heatmaps for DenseNet-121 only using Grad-CAM
    - __densenet121attA.py__: DenseNet-121-attA model and the functions to generate the heatmaps for that model
    - __densenet121attB.py__: DenseNet-121-attB model
    - __densenet_models_w_non_image.py__: standard DenseNet model + non-image feature model
    - __densenet121attA_w_non_image.py__: DenseNet-121-attA model + non-image feature model
- __resnet_models.py__: standard ResNet model from https://github.com/pytorch/vision/blob/release/0.9/torchvision/models/resnet.py

### Folder description
- labels: contains "Data_Entry_2017_v2020.csv" and the lists of training, validation and test subsets of images used in the final report
- models: contains two trained models -- DenseNet-121 and DenseNet-121-attA that can be used to generate the heatmaps
- heatmaps: output folder for the heatmaps

## Tested Environment
### Local
CPU: AMD Ryzen 5 4600H  
GPU: NV GTX 1650 / NV RTX 2060 Max-Q

### AWS
`c5` series  
`p2` series

### Dependencies

* python3.7+
* pytorch, pytorch vision, PIL, numpy, pandas, scikit-learn, matplotlib, importlib, datetime,time

## Installing

### For local
- Please install PyTorch 1.8.x via mini-conda
- If you would like to enable CUDA acceleration, install CUDA toolkit accordingly

### For AWS
- Choose `AWS Deep Learning AMI` when creating EC2 instances
- Use any venv with PyTorch 1.8.x to run the notebooks

## Authors
- [Marcus Hwai Yik Tan]
- [Xiaohan Tian]
- [Wing Chan]
- [Joshua Ceaser]
