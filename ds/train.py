import torch
import torch.optim as optim
from model import LHUNet
from utils import get_dataloaders
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
base_dir = "../ACDC"
batch_size = 8
learning_rate = 1e-4
num_epochs = 50
output_size = (224, 224)

# Load data
train_loader, val_loader, _ = get_dataloaders(base_dir, batch_size, output_size)

# Initialize model, optimizer, and loss function
model = LHUNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

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
