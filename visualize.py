import cv2
import numpy as np
from matplotlib import pyplot as plt

from classifier import *

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def slice_mask(mask_shape, slice_size, mask_arr, margin=1):
    mask = np.zeros(mask_shape)
    
    h, w = mask_shape
    h_cols = h // slice_size
    w_cols = w // slice_size
    
    for i in range(h_cols):
            for j in range(w_cols):
                mask[
                    i*slice_size+margin:(i+1)*slice_size-1-margin,
                    j*slice_size+margin:(j+1)*slice_size-1-margin
                ] = 1 if mask_arr[i*w_cols+j] else 0
                
    return mask

def slice_plots(model, device, img_path, slice_size, read_size, trans, score_thres, mask_path=None, gt_thres=0.05):
    # infer
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, read_size)
    
    scores = slice_infer(model, device, img, slice_size, trans)
    preds = [score>score_thres for score in scores]
    pred_mask = slice_mask((img.shape[0], img.shape[1]), slice_size, preds)
    
    pred_img = img.copy()
    pred_img = apply_mask(pred_img, pred_mask, (0, 0, 128))
    
    # eval
    if mask_path is not None:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, read_size)

        gts = slice_gt(mask, slice_size, gt_thres)
        results = [pred!=gt for pred, gt in zip(preds, gts)]

        gt_mask = slice_mask(mask.shape, slice_size, gts)
        result_mask = slice_mask(mask.shape, slice_size, results)

        gt_img = img.copy()
        gt_img = apply_mask(gt_img, gt_mask, (0, 128, 0))

        result_img = img.copy()
        result_img = apply_mask(result_img, result_mask, (128, 0, 0))
    
        return pred_img, gt_img, result_img
    else:
        return pred_img
