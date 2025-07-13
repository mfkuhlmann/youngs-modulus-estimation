# Youngs Modulus Estimation
 
This repository contains the supplementary code of the paper [Advances in Compliance Detection: Novel Models Using Vision-Based Tactile Sensors](https://arxiv.org/abs/2506.14980)

## Description
In this project, we built two models, LSTM-based and Transformer-based, to predict the Young’s modulus of contact objects from the RGB images of the Gelsight sensor.

## Installation
Install the requirments environment: `pip3 install -r requirements.txt`

## Dataset
The datasets can be downloaded from [here](https://huggingface.co/datasets/mburgjr/GelSight-YoungsModulus) and [here](https://people.csail.mit.edu/yuan_wz/hardnessdataset/).

They are from from the paper [Learning Object Compliance via Young’s Modulus from Single Grasps using Camera-Based Tactile Sensors](https://arxiv.org/pdf/2406.15304) and [Shape-independent hardness estimation using deep learning and a gelsight tactile sensor](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7989116url) respectively

The specific labels for the materials of the first dataset can be found in file `dataset_objects_and_compliance.csv`

## Usage
All execution parameters can be edited in `configs`

The most important one is the selection of the dataset configuration. For each dataset configuration a separate file has been defined. Using the hydra config composition feature the dataset can be set by editing the name in the config file under `defaults:` or by dynamically defining it through commandline argument.

In addition, `settings.use_optimized_architecture` needs to be set to true if the optimized version of a previous executed smac optimization should be used

### Utils
The datasets can be pre-generated using \
`python src\generate_datasets.py`
This can be useful, especially if you are planning on executing large amount of runs on a cluster.

### Optimization
The balance threshold for the balanced dataset can be optimized using \
`python src\optimize_balance_threshold.py`

The models can be optimized respectivly using \
`python src\optimize_res_tf.py` \
`python src\optimize_top10nn.py` \
`python src\optimize_vgg_lstm.py`

During the optimization the model size can get large enough to consume more then 20 GB VRam

### Train the Models
The models can be trained on the first dataset using: \
`python src\train.py`

The models can be trained on the second dataset using: \
`python src\train_combined_ds.py`

### Evaluation
The trained models can be evaluated on the first dataset using \
`python src\evaluate_folds.py`

The trained models can be evaluated on the second dataset using \
`python src\evaluate_folds_new_dataset.py`

The scatter plot can be generated using \
`python src\plot_all.py`
The checkpoints used for this need to be manually set inside the file directly

The box plot can be generated using \
`python src\create_box_plot.py` \
This script uses a manually compiled file -> will be fixed in another version

The box plot with rolling window can be generated using \
`python src\create_scatter_plot.py`
The checkpoints used for this need to be manually set inside the file directly