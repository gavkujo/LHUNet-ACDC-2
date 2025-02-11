import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice_score(pred, target):
    """
    Compute Dice Score.
    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        target (torch.Tensor): Ground truth mask.
    Returns:
        float: Dice score.
    """
    smooth = 1e-5
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def hausdorff_95(pred, target):
    """
    Compute Hausdorff 95th Percentile.
    Args:
        pred (torch.Tensor): Predicted segmentation mask.
        target (torch.Tensor): Ground truth mask.
    Returns:
        float: Hausdorff 95th percentile.
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    if np.count_nonzero(pred) == 0 or np.count_nonzero(target) == 0:
        return 0.0
    hausdorff_dist = max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])
    return np.percentile([hausdorff_dist], 95)
