import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from torchvision import transforms
from util import TwoCropTransform

import numpy as np
import os 

root_dir = r'E:\experiment\syh'
cached_data_dir = os.path.join(root_dir, 'cache')

class DataGenerator():
    
    # the fold_6 is testing set
    def __init__(self, data_dir=cached_data_dir, task3_data='task3_images_224.npy',task3_gt='labels.npy'):
        self.data_dir = data_dir
        self.task3_gt = task3_gt
        self.task3_data = task3_data
    
    def get_x_y(self):
        x = np.load(os.path.join(self.data_dir, self.task3_data)).astype('uint8')
        y = np.load(os.path.join(self.data_dir, self.task3_gt)).astype('uint8')
        return x, y
        
    def get_train_val(self, idx=1, folds=5, one_hot=False):
        '''Get training set and val set for classification network
        
        # Argment
            idx: fold for val set
            folds: number of folds except test set
        '''
        x,y = self.get_x_y()       
        val = np.load(os.path.join(self.data_dir, 'fold_%d.npy'%(idx)))
        test = np.load(os.path.join(self.data_dir, 'fold_%d.npy'%(folds+1)))
        train = ~(val|test)
        x_train = x[train]
        y_train = y[train]
        x_val = x[val]
        y_val = y[val]
        if not one_hot:
            y_train = np.argmax(y_train, -1)
            y_train = y_train.reshape(-1, 1)
            y_val = np.argmax(y_val, -1)
            y_val = y_val.reshape(-1, 1)
        return (x_train, y_train), (x_val, y_val)

    def get_test(self, folds=5, one_hot=False):
        '''Get test set for classification network
        
        # Argment
            folds: number of folds except test set
        '''  
        x,y = self.get_x_y()       
        test = np.load(os.path.join(self.data_dir, 'fold_%d.npy'%(folds+1)))
        x_test = x[test]
        y_test = y[test]
        if not one_hot:
            y_test = np.argmax(y_test, -1)
            y_test = y_test.reshape(-1, 1)
        return (x_test, y_test)
    

class GetLoader(Dataset):

    def __init__(self, data_x, data_y, transform=None):
        self.data = data_x
        self.label = data_y
        self.transform = transform
        self.size = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        if self.transform:
            data = self.transform(data)
        return data, labels

    def __len__(self):
        return self.size

    def get_classes_for_all_imgs(self):
        return self.label.reshape(-1).tolist()

def getLoader(opt):
    generator = DataGenerator()
    (train_x, train_y), (val_x, val_y) = generator.get_train_val()
    mean = (0.7570, 0.5397, 0.5641)
    std = (0.1406, 0.1521, 0.1693)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.5),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([84, 84]),
        normalize,
    ])
    train_l = GetLoader(train_x, train_y, transform=TwoCropTransform(train_transform))
    train_v = GetLoader(val_x, val_y, transform=val_transform)

    # trainLoader = DataLoader(train_l, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    # validLoader = DataLoader(train_v, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    class_count = [740, 4468, 340, 216, 732, 76, 92]
    weights = 1. / torch.Tensor(class_count)
    train_targets = train_l.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    trainLoader = DataLoader(train_l, batch_size=48, shuffle=False, drop_last=False, sampler=sampler, pin_memory=False)
    validLoader = DataLoader(train_v, batch_size=48, shuffle=False, drop_last=False, pin_memory=False)
    return trainLoader, validLoader

def getLoader_lin(opt):
    generator = DataGenerator()
    (train_x, train_y), (val_x, val_y) = generator.get_train_val()
    mean = (0.7570, 0.5397, 0.5641)
    std = (0.1406, 0.1521, 0.1693)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([84, 84]),
        normalize,
    ])

    train_l = GetLoader(train_x, train_y, transform=train_transform)
    train_v = GetLoader(val_x, val_y, transform=val_transform)

    # trainLoader = DataLoader(train_l, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    # validLoader = DataLoader(train_v, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    class_count = [740, 4468, 340, 216, 732, 76, 92]
    weights = 1. / torch.Tensor(class_count)
    train_targets = train_l.get_classes_for_all_imgs()
    samples_weights = weights[train_targets]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    trainLoader = DataLoader(train_l, batch_size=opt.batch_size, shuffle=False, drop_last=False, sampler=sampler, pin_memory=False)
    validLoader = DataLoader(train_v, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=False)
    return trainLoader, validLoader