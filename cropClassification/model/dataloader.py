from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import torch
from torchvision import transforms

class RoadsideCropImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, usage='train', mean=None, std=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels.
            root_dir (string): Directory with all the images and masks.
            usage (string): 'train' or 'val'. Determines if transformations are applied.
            mean (list, optional): Mean for normalization.
            std (list, optional): Standard deviation for normalization.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.usage = usage
        self.mean = mean if mean is not None else [0.5] * 9  # Default mean for 9 channels
        self.std = std if std is not None else [0.5] * 9    # Default std for 9 channels

        # Automatically build the transform pipeline
        self.transform = self.build_transforms()

    def build_transforms(self):
        """
        Build the transformations based on the usage (train/val).
        """
        transform_list = []

        if self.usage == 'train':
            # Apply random flip, rotation, and resized crop for training data
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))
            transform_list.append(transforms.RandomRotation(degrees=(-90, 90)))
            transform_list.append(transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)))
        elif self.usage == 'val':
            # For validation, just resize to 224x224
            transform_list.append(transforms.Resize(size=(224, 224)))

        # Add normalization (always applied)
        transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        # Return the composed transforms
        return transforms.Compose(transform_list)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Load the image and the corresponding mask (label), apply transformations if necessary,
        and return them as tensors.
        """
        # Get the image and mask paths
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['img_chip_path'])
        mask_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['lbl_chip_path'])

        # Load the multi-channel image (e.g., RGB + LAB + HSV) from a .npy file
        image = np.load(img_name)  # Assuming the image is stored as .npy with stacked channels

        # Permute the dimensions of the image to [C, H, W] format (as expected by PyTorch)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Load the mask (assuming it's a grayscale .png image)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        # Convert mask to a torch tensor
        mask = torch.from_numpy(mask).long()  # Convert mask to long tensor (for segmentation)

        # Apply the transformation pipeline (which includes normalization)
        if self.transform:
            image = self.transform(image)

        return image, mask
