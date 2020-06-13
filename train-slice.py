import os
import sys
import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import PIL
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import sklearn.metrics as metrics

from dataset import *
from resnet import *
from slice_net import *
from classifier import *

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
parser.add_argument('--attention', default='none', help='attention name')
parser.add_argument('--channel_r', default=16, type=int, help='r of channel attention')
parser.add_argument('--spartial_k', default=7, type=int, help='kernel size of spartial attention')
flags = parser.parse_args()

# consts
TRAIN_NAME = flags.train_name
TRAIN_ID = flags.train_id

# data consts
ROOT_PATH = flags.root_path
BATCH_SIZE = 8 * 4
NUM_WORKERS = 16

NUM_CLASSES = 2
INPUT_SIZE = (640, 480)
MASK_RESIZE_RATIO = 32

# trainer consts
DEVICE = flags.device
LR = flags.lr
EPOCH = flags.epochs
STEP = flags.scheduler_step
GAMMA = flags.scheduler_gamma

# model consts
ATTENTION = flags.attention
R = flags.channel_r
K = flags.spartial_k

# data
train_seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.5, iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
    )),
    iaa.OneOf([
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0.0, 1.0))),
        iaa.Sometimes(0.25, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.85, 1.3))),
    ]),
    iaa.Sometimes(0.25, iaa.Grayscale(alpha=(0.0, 1.0))),
    iaa.Resize({'height':INPUT_SIZE[1], 'width':INPUT_SIZE[0]})
])

val_seq = iaa.Sequential([
    iaa.Resize({'height':INPUT_SIZE[1], 'width':INPUT_SIZE[0]})
])

tensor_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

mask_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE[1]//MASK_RESIZE_RATIO, INPUT_SIZE[0]//MASK_RESIZE_RATIO), PIL.Image.NEAREST),
    transforms.ToTensor()
])
    
def trans(
    img, mask,
    iaa_seq=None,
    vision_trans=None,
    mask_trans=None
):
    mask_map = SegmentationMapsOnImage(mask, shape=img.shape)
    img_aug, mask_map_aug = iaa_seq(image=img, segmentation_maps=mask_map)
    
    img = vision_trans(img_aug)
    mask = mask_trans(mask_map_aug.get_arr())
    
    return img, mask

train_trans = partial(
    trans,
    iaa_seq=train_seq,
    vision_trans=tensor_trans,
    mask_trans=mask_trans
)
val_trans = partial(
    trans,
    iaa_seq=val_seq,
    vision_trans=tensor_trans,
    mask_trans=mask_trans
)

train_dataset = SegmentationDataset(
    ROOT_PATH,
    training=True,
    transform=train_trans
)
val_dataset = SegmentationDataset(
    ROOT_PATH,
    training=False,
    transform=val_trans
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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

resnet = resnet101(pretrained=True, num_classes=NUM_CLASSES)

if ATTENTION == 'se':
    attention = SE_module(resnet.feature_size, R)
elif ATTENTION == 'channel':
    self.attention = Channel_Attention(resnet.feature_size, R)
elif ATTENTION == 'spartial':
    attention = Spartial_Attention(K)
elif ATTENTION == 'cbam':
    attention = nn.Sequential(
        Channel_Attention(resnet.feature_size, R),
        Spartial_Attention(K)
    )
elif ATTENTION == 'non_local':
    attention = NonLocalBlockND(resnet.feature_size)
else:
    attention = None

model = SliceNet(resnet, num_classes=NUM_CLASSES, attention=attention)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, STEP, gamma=GAMMA, last_epoch=-1)
criterion = BitBalanceHardMiningLoss()

if __name__ == '__main__':
    train(TRAIN_NAME, TRAIN_ID, model, device, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCH, metric=metrics.jaccard_score)