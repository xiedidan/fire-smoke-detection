import os
import sys

import torch
import torch.nn as nn

from tqdm import tqdm
from tensorboardX import SummaryWriter
    
def train_epoch(epoch, model, loader, device, criterion, optimizer, scheduler, writer):
    model.train()
    
    running_loss = 0
    
    with tqdm(total=len(loader), file=sys.stdout) as pbar:
        for iter_no, (imgs, gts) in enumerate(loader):
            imgs = imgs.to(device)
            gts = gts.to(device)
            
            optimizer.zero_grad()
            
            results = model(imgs)
            losses = criterion(results, gts)
            
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            writer.add_scalar(
                'train/loss',
                losses.item(),
                epoch*len(loader)+iter_no
            )
            
            pbar.update(1)
            
    scheduler.step()
    
    return running_loss/len(loader)

def eval(epoch, model, loader, criterion, device, writer, metric=None):
    model.eval()
    
    running_loss = 0.
    running_metric = 0.
    
    with torch.no_grad():
        with tqdm(total=len(loader), file=sys.stdout) as pbar:
            for iter_no, (imgs, gts) in enumerate(loader):
                imgs = imgs.to(device)
                gts = gts.to(device)
                
                results = model(imgs)
                losses = criterion(results, gts)
                
                running_loss += losses.item()
                
                # be ware torch.max is overloaded
                preds = torch.max(nn.functional.softmax(results, dim=1), 1)[1]
                
                preds = preds.cpu().view(-1).numpy()
                gts = gts.cpu().squeeze().view(-1).numpy()
                
                if metric is not None:
                    m = metric(gts, preds)
                    running_metric += m

                pbar.update(1)
                
        if metric is not None:
            writer.add_scalar(
                'val/metric',
                running_metric/len(loader),
                epoch
            )
                
    return running_loss/len(loader)

def train(name, train_id, model, device, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, log_path='./logs', metric=None):
    writer = SummaryWriter(os.path.join(log_path, '{}_{}'.format(name, train_id)))
    
    for epoch in range(epochs):
        train_loss = train_epoch(epoch, model, train_loader, device, criterion, optimizer, scheduler, writer)
        eval_loss = eval(epoch, model, val_loader, criterion, device, writer, metric)
        
        writer.add_scalars(
            'avg/loss',
            {
                'train': train_loss,
                'val': eval_loss
            },
            epoch
        )
        
        model_path = os.path.join('./models', '{}_{}'.format(name, train_id))
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print('epoch: {}, train_loss: {}, eval_loss: {}'.format(epoch, train_loss, eval_loss))
        torch.save(model.state_dict(), os.path.join(model_path, '{:0>3d}.pth'.format(epoch)))
        
    writer.close()

# input ndarray image
def slice_infer(model, device, img, slice_size):
    model.eval()
    
    h, w, c = img.shape
    scores = []
    
    with torch.no_grad():
        for i in range(w//slice_size):
            for j in range(h//slice_size):
                data = img[
                    i*slice_size:(i+1)*slice_size-1,
                    j*slice_size:(j+1)*slice_size-1,
                    :
                ]
                data_tensor = torch.from_numpy(data).to(device)

                result = model(data_tensor)
                score = nn.functional.softmax(result, -1)[-1]
                scores.append(score.item())
            
    return scores

def slice_gt(mask, slice_size, neg_thres, pos_thres):
    h, w = mask.shape
    
    gts = []
    
    for i in range(w//slice_size):
        for j in range(h//slice_size):
            mask_slice = mask[
                i*slice_size:(i+1)*slice_size-1,
                j*slice_size:(j+1)*slice_size-1
            ]
            
            total_pixels = mask_slice.shape[0] * mask_slice.shape[1]
            pixel_count = cv2.countNonZero(mmask_slice)
            ratio = pixel_count / total_pixels
            
            if ratio < self.neg_thres:
                gts.append(False)
            elif ratio > self.pos_thres:
                gts.append(True)
                
    return gts
    
def infer(model, device, img, slice_size, score_threshold=0.5):
    scores = slice_infer(model, device, img, slice_size)
    results = torch.gt(scores, score_threshold)
    count, = torch.nonzero(results).shape
    
    return 1 if count > 0 else 0

def test(model, device, imgs, gts, slice_size, score_threshold=0.5):
    results = [infer(model, device, img.to(device), slice_size, score_threshold) for img in imgs]
    accu = metrics.accuracy_score(gts, preds)
    print(accu)
    