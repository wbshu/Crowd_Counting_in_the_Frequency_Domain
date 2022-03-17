# Crowd Counting in the Frequency Domain

This repository is the official implementation of [Crowd Counting in the Frequency Domain](https://openaccess.thecvf.com/content/CVPR2022/papers/Shu_Crowd_Counting_in_the_Frequency_Domain_CVPR_2022_paper.pdf). 

## Requirements
Python >= 3.7

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset Preparation
Download the datasets from official sites.
Group them according to official documents. 
Organize them as follows.

```
Dataset
├── ShanghaiTech
│   ├── part_A_final
|   │   ├── test_data
|   |   │   ├── ground_truth
|   |   │   │   ├── GT_IMG_1.mat 
|   |   │   │   ├── GT_IMG_2.mat
|   |   │   │   └── ... 
|   |   │   └── images
|   |   │       ├── IMG_1.jpg
|   |   │       ├── IMG_2.jpg
|   |   │       └── ... 
|   |   └── train_data
|   |       ├── ground_truth
|   |       │   ├── GT_IMG_1.mat 
|   |       │   ├── GT_IMG_2.mat
|   |       │   └── ... 
|   |       └── images
|   |           ├── IMG_1.jpg
|   |           ├── IMG_2.jpg
|   |           └── ...
|   └── part_B_final
|       ├── test_data
|       │   ├── ground_truth
|       │   │   ├── GT_IMG_1.mat 
|       │   │   ├── GT_IMG_2.mat
|       │   │   └── ... 
|       │   └── images
|       │       ├── IMG_1.jpg
|       │       ├── IMG_2.jpg
|       │       └── ... 
|       └── train_data
|           ├── ground_truth
|           │   ├── GT_IMG_1.mat 
|           │   ├── GT_IMG_2.mat
|           │   └── ... 
|           └── images
|               ├── IMG_1.jpg
|               ├── IMG_2.jpg
|               └── ...
├── UCF-QNRF
|   ├── test
|   |   ├── img_0001.jpg
|   |   ├── img_0001_ann.mat
|   |   └── ...
|   └── train
|       ├── img_0001.jpg
|       ├── img_0001_ann.mat
|       └── ...
├── jhu_crowd_v2.0
|   ├── test
|   |   ├── gt
|   |   |   ├── 0002.txt  
|   |   |   └── ...
|   |   └── images
|   |       ├── 0002.jpg
|   |       └── ...
|   ├── train
|   |   ├── gt
|   |   |   ├── 0001.txt  
|   |   |   └── ...
|   |   └── images
|   |       ├── 0001.jpg
|   |       └── ...
|   └── val
|       ├── gt
|       |   ├── 0003.txt  
|       |   └── ...
|       └── images
|           ├── 0003.jpg
|           └── ...
├── NWPU
|   ├── test
|   |   ├── 3610.jpg   
|   |   └── ...
|   ├── train
|   |   ├── 0001.jpg  
|   |   ├── 0001.mat  
|   |   └── ...   
|   └── val
|       ├── 3098.jpg
|       ├── 3098.mat
|       └── ...
└── ...

```

To prepare datasets, run this command:

```dataset
python dataset_preparation.py <arg1>
```
arg1 is selected from [SHTCA, SHTCB, QNRF, JHU++, NWPU], e.g.,

```dataset2
python dataset_preparation.py SHTCA
```

## Training
We plan to extend the paper to a journal paper. Once our journal paper is accepted,
we will release the training part. 

## Evaluation

To evaluate my model, run:

```eval
python test.py <arg1> <arg2>
```
arg1: specify the dataset name--SHTCA, SHTCB, QNRF, JHU++, NWPU, or your own prepared dataset.

arg2: the relative path of the pretrained model.

e.g.
```eval2
python test.py SHTCA Model/model_pretrain/shtca.pth
```

## Pre-trained Models

You can download my pretrained models [here](https://portland-my.sharepoint.com/:f:/g/personal/weiboshu2-c_my_cityu_edu_hk/Eor5dJSoOnRMq3CSwfbPzcwB024VVIfmn1lmD8ZOgPprHw?e=1vz2d7).
The secret code is: 7926

## Citation
If the codes help you, please cite
```citation
@inproceedings{shu2022crowd,
  title={Crowd Counting in the Frequency Domain},
  author={Shu, Weibo and Wan, Jia and Tan, Kay Chen and Kwong, Sam and Chan, Antoni B},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19618--19627},
  year={2022}
}
```

## Acknowledgement
Part of codes are from [Baysian-Crowd-Counting](https://github.com/ZhihengCV/Bayesian-Crowd-Counting).
