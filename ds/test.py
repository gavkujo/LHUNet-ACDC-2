import torch
from model import LHUNet
from utils import get_dataloaders
from metrics import dice_score, hausdorff_95

# Load model
model = LHUNet()
model.load_state_dict(torch.load("lhunet.pth"))
model.eval()

# Load test data
_, _, test_loader = get_dataloaders("data", batch_size=8)

# Evaluate
dice_scores = []
hausdorff_scores = []

for batch in test_loader:
    x, y = batch
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1)
        dice = dice_score(pred, y)
        hausdorff = hausdorff_95(pred, y)
        dice_scores.append(dice)
        hausdorff_scores.append(hausdorff)

print(f"Average Dice Score: {sum(dice_scores) / len(dice_scores)}")
print(f"Average Hausdorff 95th Percentile: {sum(hausdorff_scores) / len(hausdorff_scores)}")
