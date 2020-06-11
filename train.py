import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dataset import *
from resnet import *
from classifier import *
from sampler import BalancedStatisticSampler

# args
parser = argparse.ArgumentParser(description='Slice Classifier Training')
parser.add_argument('--device', default='cuda', help='device (cuda / cpu)')
parser.add_argument('--root_path', default='./data/fire', help='dataset root path')
parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
parser.add_argument('--scheduler_step', default=15, type=int, help='epoch to reduce learning rate')
parser.add_argument('--scheduler_gamma', default=0.1, type=float, help='step scheduler gamma')
parser.add_argument('--train_name', default='slice', help='train name')
parser.add_argument('--train_id', default='01', help='train id')
parser.add_argument('--epochs', default=20, type=int, help='epoch to train')
parser.add_argument('--slice_size', default=40, type=int, help='square slice size')
parser.add_argument('--pos_thres', default=0.1, type=float, help='positive slice threshold')
parser.add_argument('--neg_thres', default=0.01, type=float, help='negative slice threshold')
flags = parser.parse_args()

# consts
TRAIN_NAME = flags.train_name
TRAIN_ID = flags.train_id

POS_THRESHOLD = flags.pos_thres
NEG_THRESHOLD = flags.neg_thres

# data consts
ROOT_PATH = flags.root_path
NUM_CLASSES = 2 # fg + 1(bg)
READ_SIZE = (640, 480)
SLICE_SIZE = flags.slice_size
INPUT_SIZE = 224
BATCH_SIZE = 64 * 4
NUM_WORKERS = 16

# trainer consts
DEVICE = flags.device
LR = flags.lr
EPOCH = flags.epochs
STEP = flags.scheduler_step
GAMMA = flags.scheduler_gamma

# data
train_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
val_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = SegSliceDataset(
    ROOT_PATH,
    slice_size=SLICE_SIZE,
    read_size=READ_SIZE,
    training=True,
    transform=train_trans,
    pos_thres=POS_THRESHOLD,
    neg_thres=NEG_THRESHOLD
)
val_dataset = SegSliceDataset(
    ROOT_PATH,
    slice_size=SLICE_SIZE,
    read_size=READ_SIZE,
    training=False,
    transform=val_trans,
    pos_thres=POS_THRESHOLD,
    neg_thres=NEG_THRESHOLD
)

train_sampler = BalancedStatisticSampler(
    train_dataset.ys,
    NUM_CLASSES,
    BATCH_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# model
device = torch.device(DEVICE)

model = resnet101(pretrained=True, num_classes=NUM_CLASSES)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, STEP, gamma=GAMMA, last_epoch=-1)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    train(TRAIN_NAME, TRAIN_ID, model, device, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCH)