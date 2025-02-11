import torch
import torch.optim as optim
from model import LHUNet
from utils import get_dataloaders

# Hyperparameters
data_dir = "data"
batch_size = 8
learning_rate = 1e-4
num_epochs = 50

# Load data
train_loader, val_loader = get_dataloaders(data_dir, batch_size)

# Initialize model, optimizer, and loss function
model = LHUNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = lambda pred, target: dice_loss(pred, target) + nn.CrossEntropyLoss()(pred, target)

def dice_loss(pred, target):
    smooth = 1e-5
    pred = torch.softmax(pred, dim=1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        x, y = batch
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "lhunet.pth")
