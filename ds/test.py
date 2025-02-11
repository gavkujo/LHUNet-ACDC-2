import torch
from model import LHUNet
from utils import get_dataloaders

# Load model
model = LHUNet()
model.load_state_dict(torch.load("lhunet.pth"))
model.eval()

# Load test data
_, _, test_loader = get_dataloaders("data", batch_size=8)

# Evaluate
dice_scores = []
for batch in test_loader:
    x, y = batch
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1)
        dice = dice_score(pred, y)
        dice_scores.append(dice)

print(f"Average Dice Score: {sum(dice_scores) / len(dice_scores)}")

def dice_score(pred, target):
    intersection = (pred == target).sum()
    return (2 * intersection) / (pred.sum() + target.sum())
