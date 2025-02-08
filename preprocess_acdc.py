import os
import numpy as np
import cv2
import nibabel as nib
import glob
import torch
from torchvision import transforms
from tqdm import tqdm

# Paths
DATASET_PATH = "path_to_acdc_dataset"
OUTPUT_PATH = "preprocessed_acdc"

# Create output directories
os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "masks"), exist_ok=True)

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Resize to match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

def preprocess_acdc():
    img_paths = sorted(glob.glob(os.path.join(DATASET_PATH, "imagesTr", "*.nii.gz")))
    mask_paths = sorted(glob.glob(os.path.join(DATASET_PATH, "labelsTr", "*.nii.gz")))

    for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
        # Load NIfTI images
        img_nifti = nib.load(img_path)
        mask_nifti = nib.load(mask_path)

        img_data = img_nifti.get_fdata()
        mask_data = mask_nifti.get_fdata()

        # Process each slice
        for slice_idx in range(img_data.shape[-1]):
            img_slice = img_data[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx]

            # Convert to uint8 for saving
            img_slice = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask_slice = (mask_slice > 0).astype(np.uint8) * 255  # Binarize mask

            # Apply transformations
            img_tensor = transform(img_slice)
            mask_tensor = transform(mask_slice)

            # Save
            img_save_path = os.path.join(OUTPUT_PATH, "images", f"{os.path.basename(img_path)}_{slice_idx}.png")
            mask_save_path = os.path.join(OUTPUT_PATH, "masks", f"{os.path.basename(mask_path)}_{slice_idx}.png")

            cv2.imwrite(img_save_path, img_slice)
            cv2.imwrite(mask_save_path, mask_slice)

if __name__ == "__main__":
    preprocess_acdc()
