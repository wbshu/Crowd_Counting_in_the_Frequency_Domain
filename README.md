# Crowd Counting in the Frequency Domain

This repository is the official implementation of [Crowd Counting in the Frequency Domain](https://openaccess.thecvf.com/content/CVPR2022/papers/Shu_Crowd_Counting_in_the_Frequency_Domain_CVPR_2022_paper.pdf) and `Generalized Characteristic Function Loss for Crowd Analysis in the
Frequency Domain' (appear on TPAMI 2024)

## Some notes

1, For using the noisy crowd counting loss, please see the comments at trainer.py/Chf_trainer class//init method. Generally
speaking, the noisy crowd counting loss performs better then general chf loss, but it also depends on the dataset and the 
backbone network. In practical application, I suggest you to try both of them. 

2, For crowd localization part, I may release them in the future. Since the codes are still remained to be collated now, but I'm 
busy in my graduation currently. 

3, For the transformer based network, the loss performs very differently on different transformer-based networks, I guess 
there maybe some overfitting problems on some special transformer structures. You can try it on diverse transformer structure
and find the problem. I'm sure that this loss is powerful, but there is still much improvement space, I'm glad to see any
improvement on it.  

4, If you use the codes for academic purpose, please cite my papers properly. To fast understand my codes, I suggest you
to read the comments in my codes. Hope that my codes can help you solve some problems. Have fun!

5, For the supplementary material of `Generalized Characteristic Function Loss for Crowd Analysis in the Frequency Domain',
you can download it in this page: http://visal.cs.cityu.edu.hk/publications/ (seach the title in the page and download 
the supplemental)

## Requirements
Python >= 3.7

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset Preparation
Download the datasets from official sites.
Split them according to official documents. 
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
To train the model in the paper, run this command:

```train
python train.py <arg1> <arg2>
```
arg1: specify the dataset name--SHTCA, SHTCB, QNRF, JHU++, NWPU, or your own prepared dataset.

arg2: give a name to distinguish the best model of this trial from others (you don't need to add `.pth', just give the
file name without suffix).

e.g.
```train2
python train.py SHTCA best_model
```
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

@article{shu2023generalized,
  title={Generalized Characteristic Function Loss for Crowd Analysis in the Frequency Domain},
  author={Shu, Weibo and Wan, Jia and Chan, Antoni B},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgement
Part of codes are from [Baysian-Crowd-Counting](https://github.com/ZhihengCV/Bayesian-Crowd-Counting).
