# Part-based Graph Convolutional Network for Skeleton-based Action Recognition

Official repository for the code from BMVC (British Machine Vision Conference) paper "[Part-based Graph Convolutional Network for Action Recognition](http://bmvc2018.org/contents/papers/1003.pdf)". The implementation is done in Pytorch and works on it's recent stable version. The repository includes:

- [x] Code for the final model used in the paper.
- [x] Model checkpoints for model trained on NTURGB+D Cross Subject and Cross View data splits.
- [x] Training and testing config as well as data preparation scripts for NTURGB+D dataset.
- [x] Training config as well as data preparation scripts for HDM05 dataset.

TODOs:

- [ ] Code for visualizing results.
- [ ] Document how to extend the code for other projects.

## Getting Started

1. Download the NTURGB+D dataset (with 60 action classes) following this [link](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). Unzip the archive and store all the skeleton files in a single directory:

```
unzip nturgbd_skeletons_s001_to_s017.zip -d nturgb+d_skeletons
```

2. Clone the repository:

```
git clone https://github.com/kalpitthakkar/pb-gcn.git
```

3. Download the pretrained model checkpoints. To download the checkpoints for both the cross-subject and cross-view splits:

```
bash download_checkpoints.sh <path_to_download_directory>
```

## Training / Testing Models

1. First of all, we need to define the required configuration variables in the YAML file. The instructions on it's structure and editing the file are [here](https://github.com/kalpitthakkar/pb-gcn/tree/master/config/README.md).

2. Once the configuration file is ready, you can start training the model:

```
python run.py --config <path_to_YAML_config_file>
```

## Citation

For citing our paper:

```
@article{thakkar2018part,
title={Part-based Graph Convolutional Network for Action Recognition},
author={Thakkar, Kalpit and Narayanan, PJ},
journal={arXiv preprint arXiv:1809.04983},
year={2018}
}
```
