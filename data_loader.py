import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from PIL import Image
import pandas as pd
import os
import random
from glob import glob


def get_cifar100_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True):
    
    # define transforms
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load dataset
    dataset = datasets.CIFAR100(root=data_dir,
                                transform=trans,
                                download=True,
                                train=True)
    if shuffle:
        np.random.seed(random_seed)
    
    torch.manual_seed(101)
    
    # V1
    is_iid, pnumber = True, 2
    lst = []
    class_size = len(dataset) // len(set(dataset.targets))
    num_classes = len(set(dataset.targets))

    # dictionary of labels map 
    labels = dataset.targets
    dct = {}
    for idx, label in enumerate(labels):
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp
        
    # probabilities
    torch.set_printoptions(precision=2)
    probs = []
    for i in range(num_classes):
        if is_iid:
            probs.append([1.0 / pnumber] * pnumber)
        else:
            rand = torch.rand(pnumber)
            prob = rand / sum(rand)
            probs.append(prob)
    print(probs, end="\n\n")

    # division
    lst = {i: [] for i in range(pnumber)}
    for class_id, distribution in enumerate(probs):
        from_id = 0
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:]
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id

    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=True) for i in range(pnumber)]

    counts = [0] * 100
    for label in subsets[0]:
        counts[label[1]] += 1
    print('1st set: ', counts, sum(counts), end="\n\n")

    counts = [0] * 100
    for label in subsets[1]:
        counts[label[1]] += 1
    print('2nd set: ', counts, sum(counts), end="\n\n")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )
    return t_loaders + [train_loader]


def get_cifar100_test_loader(data_dir, batch_size, num_workers=4, pin_memory=True):
    
    # define transforms
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load dataset
    dataset = datasets.CIFAR100(
        data_dir, train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def get_cifar10_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True):
    
    # define transforms
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize([0.4915, 0.4823, .4468], [0.2470, 0.2435, 0.2616])
    ])

    # load dataset
    dataset = datasets.CIFAR10(root=data_dir,
                               transform=trans,
                               download=True,
                               train=True)
    if shuffle:
        np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    is_iid, pnumber = True, 5

    lst = []
    class_size = len(dataset) // len(set(dataset.targets))
    num_classes = len(set(dataset.targets))

    # dictionary of labels map
    labels = dataset.targets
    dct = {}
    for idx, label in enumerate(labels):
        if label not in dct:
            dct[label] = []
        dct[label].append(idx)

    for i in range(num_classes):
        temp = random.sample(dct[i], len(dct[i]))
        dct[i] = temp
        
    # probabilities
    torch.set_printoptions(precision=3)
    probs = []
    for i in range(num_classes):
        if is_iid:
            probs.append([1.0 / pnumber] * pnumber)
        else:
            rand = torch.rand(pnumber)
            prob = rand / sum(rand)
            probs.append(prob)
    print(probs, end="\n\n")

    # division
    lst = {i: [] for i in range(pnumber)}
    for class_id, distribution in enumerate(probs):
        from_id = 0
        for participant_id, prob in enumerate(distribution):
            to_id = int(from_id + prob * class_size)
            if participant_id == pnumber - 1:
                lst[participant_id] += dct[class_id][from_id:]
            else:
                lst[participant_id] += dct[class_id][from_id:to_id]
            from_id = to_id

    subsets = [torch.utils.data.Subset(dataset, lst[i]) for i in range(pnumber)]
    t_loaders = [torch.utils.data.DataLoader(subsets[i], batch_size=batch_size, shuffle=True) for i in range(pnumber)]

    for pi in range(pnumber):
        counts = [0] * 10
        for label in subsets[pi]:
            counts[label[1]] += 1
        print(f'{pi+1} set: ', counts, sum(counts), end="\n")

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [train_loader]  # 5/10 participants # train_loader


def get_cifar10_test_loader(data_dir, batch_size, num_workers=4, pin_memory=True):

    # define transforms
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4915, 0.4823, .4468], [0.2470, 0.2435, 0.2616])
    ])

    # load dataset
    dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=trans
    )

    np.random.seed(101)
    torch.manual_seed(101)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def get_mnist_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True):
    # define transforms
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load dataset
    dataset = datasets.MNIST(root=data_dir, transform=trans, download=True, train=True)
    if shuffle:
        np.random.seed(random_seed)

    is_iid, pnumber = True, 2
    if is_iid:
        coefs = [0.8, 0.75, 2 / 3, 0.5] if pnumber == 5 else [0.5]
        lst = []
        temp = dataset 
        while len(lst) < pnumber:
            labels = [i[1] for i in temp]
            t1_set, t2_set = train_test_split(
                temp,
                test_size=coefs[len(lst)],
                random_state=101,
                stratify=labels
            )
            lst.append(t1_set)
            if len(lst) == pnumber - 1:
                lst.append(t2_set)
            temp = t2_set

        counts = [0] * 10
        for label in lst[0]:
            counts[label[1]] += 1
        print('first set: ', counts, end="\n\n")

        t_loaders = [torch.utils.data.DataLoader(elem, batch_size=batch_size, shuffle=shuffle, 
                        num_workers=num_workers, pin_memory=pin_memory) for elem in lst]


    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return t_loaders + [train_loader]


def get_mnist_test_loader(data_dir, batch_size, num_workers=4, pin_memory=True):
    # define transforms
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return(len(self.df))

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y


def get_ham10000_train_loader(data_dir, batch_size, random_seed, shuffle=True, num_workers=4, pin_memory=True):
    
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    
    # ----
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    # extracts the image id to match it with the .csv label file
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0] : x for x in all_image_path}
    lesion_type_dict = {'nv': 'Melanocytic nevi', 'mel': 'dermatofibroma', 'bkl': 'Benign keratosis-like lesions ', 
                        'bcc': 'Basal cell carcinoma', 'akiec': 'Actinic keratoses', 'vasc': 'Vascular lesions', 
                        'df': 'Dermatofibroma'
                       }
    input_size = 224
    
    # ----
    df = pd.read_csv(os.path.join(data_dir,'HAM10000_metadata.csv'))
    df['path'] = df['image_id'].map(imageid_path_dict.get) 
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
    
    # finding number of images in each group
    ndf = df.groupby('lesion_id').count()
    # finding out lesion id that have only one image
    ndf = ndf[ndf['image_id']==1]
    ndf.reset_index(inplace=True)
    
    # identify ones with duplicate images and only one image
    def get_duplicate(x):
        uniq = list(ndf['lesion_id'])
        if x in uniq:
            return 'unduplicate'
        return 'duplicated'

    # new column of lesion id
    df['duplicates'] = df['lesion_id']
    # applying function to this column
    df['duplicates'] = df['duplicates'].apply(get_duplicate)
    
    # filtering images which are not duplicated
    df_undup = df[df['duplicates']=='unduplicate']
    
    # creating validation set
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.20,random_state=101, stratify=y)
    
    # creating training set on df (including duplicates)
    # Function identifies if an image is part of the train or val set.
    def get_val_rows(x):
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        return 'train'
    
    # applying it
    df['train_or_val'] = df['image_id']
    df['train_or_val'] = df['train_or_val'].apply(get_val_rows)
    #filter out train rows
    df_train = df[df['train_or_val']=='train']
    
    # # creating copies to balance
    # data_aug_rate = [15,10,5,50,0,40,5]
    # for i in range(7):
    #     if data_aug_rate[i] > 0:
    #         df_train = df_train.append([df_train.loc[df_train['cell_type_idx']==i,:]]*(data_aug_rate[i]-1),ignore_index=True)
            
    df_train = df_train.reset_index()
        
    # if not training from scratch and using pretrained
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    
    # data augmentation
    train_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomRotation(20),
                                         transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    
    # dataset
    dataset = HAM10000(df_train, transform=train_transform)
    
    is_iid, pnumber = True, 2
    if is_iid:
        coefs = [0.8, 0.75, 2 / 3, 0.5] if pnumber == 5 else [0.5]
        lst = []
        temp = dataset 
        while len(lst) < pnumber:
            t1_set, t2_set = train_test_split(
                temp,
                test_size=coefs[len(lst)],
                random_state=101 
            )
            lst.append(t1_set)
            if len(lst) == pnumber - 1:
                lst.append(t2_set)
            temp = t2_set
        counts = [0] * 7
        for label in lst[0]:
            counts[label[1]] += 1
        print('first set: ', counts, end="\n\n")
        
        t_loaders = [torch.utils.data.DataLoader(elem, batch_size=batch_size, shuffle=shuffle, 
                        num_workers=num_workers, pin_memory=pin_memory) for elem in lst]
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )
    
    return t_loaders + [train_loader]


def get_ham10000_test_loader(data_dir, batch_size, num_workers=4, pin_memory=True):
    
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0] : x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    input_size = 224
    
    # ----
    df = pd.read_csv(os.path.join(data_dir,'HAM10000_metadata.csv'))
    df['path'] = df['image_id'].map(imageid_path_dict.get) 
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
    
    ndf = df.groupby('lesion_id').count()
    ndf = ndf[ndf['image_id']==1]
    ndf.reset_index(inplace=True)
    
    # identify ones with duplicate images and only one image
    def get_duplicate(x):
        uniq = list(ndf['lesion_id'])
        if x in uniq:
            return 'unduplicate'
        return 'duplicated'

    df['duplicates'] = df['lesion_id']
    df['duplicates'] = df['duplicates'].apply(get_duplicate)
    df_undup = df[df['duplicates']=='unduplicate']
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.20,random_state=101, stratify=y)
    df_val=df_val.reset_index()

    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)    
    val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean,std)])
    
    # dataset
    dataset = HAM10000(df_val, transform=val_transform)

    # dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    
    return data_loader
