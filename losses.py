import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)[:, 1]
        target = target.float()
        intersection = (pred * target).sum(dim=(1,2))
        dice = (2*intersection + self.smooth) / (pred.sum(dim=(1,2)) + target.sum(dim=(1,2)) + self.smooth)
        return 1 - dice.mean()

class HybridLoss(nn.Module):
    def __init__(self, weights=(0.4, 0.6), ds_weight=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w = weights
        self.ds_w = ds_weight

    def forward(self, pred, ds_preds, target):
        main_loss = self.w[0]*self.ce(pred, target) + self.w[1]*self.dice(pred, target)
        ds_loss = 0
        for ds in ds_preds:
            ds_resized = F.interpolate(ds, size=target.shape[1:], mode='bilinear', align_corners=True)
            ds_loss += 0.5*self.ce(ds_resized, target) + 0.5*self.dice(ds_resized, target)
        ds_loss /= len(ds_preds)
        return main_loss + self.ds_w * ds_loss

def compute_metrics(pred_mask, true_mask, threshold=0.5):
    pred = (pred_mask > threshold).astype(np.float32)
    true = true_mask.astype(np.float32)
    
    tp = (pred * true).sum()
    fp = (pred * (1 - true)).sum()
    fn = ((1 - pred) * true).sum()
    
    smooth = 1e-6
    dice = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    iou  = (tp + smooth) / (tp + fp + fn + smooth)
    prec = (tp + smooth) / (tp + fp + smooth)
    rec  = (tp + smooth) / (tp + fn + smooth)
    f1   = 2 * prec * rec / (prec + rec + smooth)
    return {'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1}