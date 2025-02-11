import os
import re
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom

class ACDCDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None, output_size=(256, 256)):
        """
        ACDC Dataset Loader for LHU-Net
        - Loads 2D slices for training.
        - Loads full volumes for validation/testing.
        """
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.output_size = output_size

        train_ids, val_ids, test_ids = self._get_ids()
        
        if split == "train":
            data_dir = os.path.join(base_dir, "ACDC_training_slices")
            patient_ids = train_ids
        else:  # Validation or Test
            data_dir = os.path.join(base_dir, "ACDC_training_volumes")
            patient_ids = val_ids if split == "val" else test_ids

        self.sample_list = [
            f for f in os.listdir(data_dir) if any(re.match(f"{pid}.*", f) for pid in patient_ids)
        ]
        print(f"Total {split} samples: {len(self.sample_list)}")

    def _get_ids(self):
        """Defines train, val, and test patient splits."""
        all_cases = [f"patient{str(i).zfill(3)}" for i in range(1, 101)]
        test_cases = all_cases[:20]  # First 20 patients for testing
        val_cases = all_cases[20:30]  # Next 10 for validation
        train_cases = all_cases[30:]  # Remaining 70 for training
        return train_cases, val_cases, test_cases

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        filename = self.sample_list[idx]
        file_path = os.path.join(self.base_dir, 
                                 "ACDC_training_slices" if self.split == "train" else "ACDC_training_volumes",
                                 filename)
        
        with h5py.File(file_path, "r") as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:]
        
        if self.split == "train" and self.transform:
            image, label = self.transform(image, label)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(label, dtype=torch.long)  # Segmentation mask

        return {"image": image, "label": label, "case_name": filename.replace(".h5", "")}

# Apply augmentation during training
def random_augment(image, label):
    """Apply random rotations, flips, and rescaling."""
    if random.random() > 0.5:
        k = np.random.randint(0, 4)
        image, label = np.rot90(image, k), np.rot90(label, k)

    if random.random() > 0.5:
        axis = np.random.randint(0, 2)
        image, label = np.flip(image, axis), np.flip(label, axis)

    return image, label
