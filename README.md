# TDC
This repository is the implementation of "Infer from What You Have Seen Before: Temporally-dependent Classifier for Semi-supervised Video Segmentation" (Accepted by CVPR 2024). It is designed for semi-supervised video semantic segmentation task.

## Install & Requirements
The code has been tested on pytorch=1.8.2 and python3.8.

**To Install python packages**
```
pip install -r requirements.txt
```

## Download Pretrained Weights
````bash
mkdir ./TDC/pretrained
cd ./TDC/pretrained
# download resnet18 imagenet pretrained weight
wget http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth
````

## Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) datasets.

Your directory tree should be look like this:
````bash
./TDC/data
├── cityscapes
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
````

## Prepare Downsample Dataset
Generated downsample dataset would be saved in ./data
````bash
cd ./TDC
python tools/data_downsample.py
````

## Stage One Training of Accel
For example, train image segmentation model on 2 GPUs. Checkpoints would be saved in ./TDC/work_dirs.
````bash
# train PSP18 TDC model
cd ./TDC/exp/TDC_30_res18/scripts
bash train.sh
````

## Stage Two Training of Accel
Please refer to [IFR](https://github.com/jfzhuang/IFR) repo.