import torch
import numpy as np
import cv2
import glob
import os
from model import LHU_Net  # Import LHU-Net model
from acdc_dataloader import get_dataloader
from medpy.metric import binary

# Load model
MODEL_PATH = "lhunet_acdc.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = LHU_Net().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Dice Score Function
def dice_score(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    return 2 * (pred & target).sum() / (pred.sum() + target.sum() + 1e-8)

# Hausdorff Distance Function
def hausdorff_distance(pred, target):
    return binary.hd(pred, target)

# Evaluate model
def evaluate():
    dataloader = get_dataloader(batch_size=1, shuffle=False)
    dice_scores = []
    hd_distances = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs).cpu().numpy()
            masks = masks.cpu().numpy()

            # Compute metrics
            dice = dice_score(outputs > 0.5, masks > 0.5)
            hd = hausdorff_distance(outputs > 0.5, masks > 0.5)

            dice_scores.append(dice)
            hd_distances.append(hd)

    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Mean Hausdorff Distance: {np.mean(hd_distances):.4f}")

if __name__ == "__main__":
    evaluate()
