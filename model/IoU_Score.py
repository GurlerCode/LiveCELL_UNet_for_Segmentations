import torch

# Function to calculate Intersection over Union (IoU)
def iou_score(pred, target):
    intersection = torch.logical_and(pred, target).sum((1, 2))
    union = torch.logical_or(pred, target).sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()