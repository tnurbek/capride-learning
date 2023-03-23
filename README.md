# CaPriDe Learning

This repository is an implementation of CVPR 2023 paper titled: "CaPriDe Learning: Confidential and Private Decentralized Learning based on Encryption-friendly Distillation Loss". 

## Dependencies
```
pip install -r requirements.txt
```

## Run CaPriDe Learning
To train 5 models in CaPriDe learning protocol: (25 epochs correspond to the number of local training epochs.)
Default dataset: CIFAR-10 (Homogeneous setting). To set the data setting to heterogeneous, simply change `data_loader.py` file, `get_cifar10_train_loader()` function. 

```
python3 main.py --model_num 5 --is_train 1 --init_lr 0.1 --gamma 0.1 --use_gpu 1 --epochs 25 --resume 0 --save_name capride_cifar10_iid_p5_model 
```

## Datasets
CIFAR-10 and CIFAR-100 datasets will be downloaded directly from `torchvision`. 

Download HAM10000 dataset using this [URL Link](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000). 

## Encrypted Inference 
To enable FHE scheme, refer to [this link](https://github.com/IBM/fhe-toolkit-linux). 
To install it, you need to have `Linux based Docker container` (as a programming language you can choose either `Python` or `C++`). 