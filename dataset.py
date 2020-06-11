import os
import sys
import random
import multiprocessing
from multiprocessing import Pool
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# dataset
class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_path,
        training=True,
        image_ext='.png',
        test_ratio=0.1,
        transform=None
    ):
        self.root_path = root_path
        self.training = training
        self.transform = transform 
        
        # train / val split per image
        filenames = os.listdir(os.path.join(root_path, 'images'))
        img_filenames = []
        
        for filename in filenames:
            if image_ext in filename:
                img_filenames.append(filename)
                
        self.img_filenames = img_filenames
        
        random.seed(0)
        train_indices = random.sample(
            list(range(len(img_filenames))),
            int(len(img_filenames)*(1.-test_ratio))
        )
        
        if self.training:
            self.indices = train_indices
        else:
            self.indices = []
        
            for i in range(len(img_filenames)):
                if i not in train_indices:
                    self.indices.append(i)
                    
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        img_file = os.path.join(
            self.root_path,
            'images',
            self.img_filenames[self.indices[index]]
        )
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_file = os.path.join(
            self.root_path,
            'masks',
            self.img_filenames[self.indices[index]]
        )
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        return img, mask

# put png images under root_path/images, and masks under root_path/masks
class SegSliceDataset(Dataset):
    def __init__(
        self,
        root_path,
        slice_size=40,
        read_size=(640,480),
        training=True,
        transform=None,
        image_ext='.png',
        test_ratio=0.1,
        pos_thres=0.1,
        neg_thres=0.01
    ):
        self.root_path = root_path
        self.training = training
        self.transform = transform
        self.slice_size = slice_size
        self.read_size = read_size
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.num_classes = 2
        
        # train / val split per image
        filenames = os.listdir(os.path.join(root_path, 'images'))
        img_filenames = []
        
        for filename in filenames:
            if image_ext in filename:
                img_filenames.append(filename)
                
        self.img_filenames = img_filenames
        
        random.seed(0)
        train_indices = random.sample(list(range(len(img_filenames))), int(len(img_filenames)*(1.-test_ratio)))
        
        if self.training:
            self.indices = train_indices
        else:
            self.indices = []
        
            for i in range(len(img_filenames)):
                if i not in train_indices:
                    self.indices.append(i)
                    
        # create slices
        self.xs = []
        self.ys = []
        
        with tqdm(total=len(self.indices), file=sys.stdout) as pbar:
            for i, file_index in enumerate(self.indices):
                xs, ys = self._split_img(
                    os.path.join(root_path, 'images', img_filenames[file_index]),
                    os.path.join(root_path, 'masks', img_filenames[file_index])
                )
                self.xs.extend(xs)
                self.ys.extend(ys)
                
                pbar.update(1)
        
    def _split_img(self, img_path, mask_path):        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.read_size)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, self.read_size)

        h, w = mask.shape
        
        imgs = []
        masks = []
        
        for i in range(h//self.slice_size):
            for j in range(w//self.slice_size):
                imgs.append(img[
                    i*self.slice_size:(i+1)*self.slice_size-1,
                    j*self.slice_size:(j+1)*self.slice_size-1,
                    :
                ])
                masks.append(mask[
                    i*self.slice_size:(i+1)*self.slice_size-1,
                    j*self.slice_size:(j+1)*self.slice_size-1
                ])
                # print(i, j, masks[-1].shape)
        
        xs = []
        ys = []
        
        for i in range(len(masks)):
            total_pixels = masks[i].shape[0] * masks[i].shape[1]
            pixel_count = cv2.countNonZero(masks[i])
            ratio = pixel_count / total_pixels
            
            if ratio < self.neg_thres:
                xs.append(imgs[i])
                ys.append(0)
            elif ratio > self.pos_thres:
                xs.append(imgs[i])
                ys.append(1)
                
        return xs, ys
        
    def __len__(self):
        return len(self.ys)
        
    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(self.xs[index])
            
        return img, self.ys[index]