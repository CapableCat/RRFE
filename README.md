# RRFE
PyTorch's implementation of "Representation Robustness and Feature Expansion for Exemplar-Free Class Incremental Learning".

# Requirements
+ Python 3.8
+ PyTorch 1.8.1 (>1.1.0)
+ cuda 11.3

# Preparing Datasets
Download following datasets:
> CIFAR-100

> Tiny-ImageNet

> ImageNet

Place the dataset in the data_manager/dataset folder.

# Trainning
+ CIFAR100

`python main.py --confg ./exps/rrfe_cifar100.json`

+ TinyImageNet

'python main.py --config ./exps/rrfe_tiny.json'

+ ImageNet-subset

'python main.py --config ./exps/rrfe_imagenet.json'
