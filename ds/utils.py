import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from scipy.ndimage import zoom
import random

class ACDCDataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        self._load_data()

    def _load_data(self):
        train_ids, val_ids, test_ids = self._get_ids()
        if self.split == 'train':
            data_dir = os.path.join(self.base_dir, "ACDC_training_slices")
            for patient_id in train_ids:
                patient_files = [f for f in os.listdir(data_dir) if f.startswith(patient_id)]
                self.sample_list.extend([os.path.join(data_dir, f) for f in patient_files])
        elif self.split == 'val':
            data_dir = os.path.join(self.base_dir, "ACDC_training_volumes")
            for patient_id in val_ids:
                patient_files = [f for f in os.listdir(data_dir) if f.startswith(patient_id)]
                self.sample_list.extend([os.path.join(data_dir, f) for f in patient_files])
        elif self.split == 'test':
            data_dir = os.path.join(self.base_dir, "ACDC_training_volumes")
            for patient_id in test_ids:
                patient_files = [f for f in os.listdir(data_dir) if f.startswith(patient_id)]
                self.sample_list.extend([os.path.join(data_dir, f) for f in patient_files])

    def _get_ids(self):
        all_cases = ["patient{:0>3}".format(i) for i in range(1, 101)]
        test_ids = ["patient{:0>3}".format(i) for i in range(1, 21)]
        val_ids = ["patient{:0>3}".format(i) for i in range(21, 31)]
        train_ids = [i for i in all_cases if i not in test_ids + val_ids]
        return train_ids, val_ids, test_ids

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        file_path = self.sample_list[idx]
        with h5py.File(file_path, 'r') as f:
            image = np.array(f['image'][:])
            label = np.array(f['label'][:])

        if self.transform:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            image, label = sample['image'], sample['label']

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        return image, label

class RandomGenerator:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = self.random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = self.random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        return {'image': image, 'label': label}

    def random_rot_flip(self, image, label):
        k = random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label

    def random_rotate(self, image, label):
        angle = random.randint(-20, 20)
        image = ndimage.rotate(image, angle, reshape=False)
        label = ndimage.rotate(label, angle, reshape=False)
        return image, label

def get_dataloaders(base_dir, batch_size=8, output_size=(224, 224)):
    train_dataset = ACDCDataset(base_dir, split='train', transform=RandomGenerator(output_size))
    val_dataset = ACDCDataset(base_dir, split='val', transform=RandomGenerator(output_size))
    test_dataset = ACDCDataset(base_dir, split='test', transform=RandomGenerator(output_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
