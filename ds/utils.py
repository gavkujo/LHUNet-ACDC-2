import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, AddChannel, ScaleIntensity, RandRotate, RandFlip, ToTensor

class ACDCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.patient_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

    def __len__(self):
        return len(self.patient_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.patient_files[idx])
        with h5py.File(file_path, 'r') as f:
            image = np.array(f['image'][:])
            mask = np.array(f['mask'][:])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_transforms():
    return Compose([
        AddChannel(),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5),
        RandFlip(spatial_axis=0, prob=0.5),
        ToTensor()
    ])

def get_dataloaders(data_dir, batch_size=8, split_ratio=0.8):
    dataset = ACDCDataset(data_dir, transform=get_transforms())
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
