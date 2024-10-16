import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class RoadsideCropImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, usage='train', mean=None, std=None, use_ancillary=False, ancillary_classes=3):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths, labels, and ancillary data.
            root_dir (string): Directory with all the images and masks.
            usage (string): 'train' or 'val'. Determines if transformations are applied.
            mean (list, optional): Mean for normalization.
            std (list, optional): Standard deviation for normalization.
            use_ancillary (bool, optional): Whether to include ancillary data or not.
            ancillary_classes (int): Number of classes in the ancillary data (for one-hot encoding).
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.usage = usage
        self.mean = mean if mean is not None else [0.5] * 9  # Default mean for 9 channels
        self.std = std if std is not None else [0.5] * 9    # Default std for 9 channels
        self.use_ancillary = use_ancillary
        self.ancillary_classes = ancillary_classes  # Number of classes in the ancillary data

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

    def one_hot_encode(self, label):
        """
        One-hot encode the given label based on the number of ancillary classes.

        Args:
            label (int): The categorical label to encode.

        Returns:
            torch.Tensor: One-hot encoded vector.
        """
        return F.one_hot(torch.tensor(label), num_classes=self.ancillary_classes).float()

    def __len__(self):
        """Return the total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Load the image, mask (label), and optional ancillary data, apply transformations if necessary,
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

        # Check if ancillary data should be included
        if self.use_ancillary:
            # Load ancillary data from the dataframe
            # Assuming the ancillary data is stored as a categorical integer (e.g., 0, 1, 2)
            ancillary_label = self.dataframe.iloc[idx]['crop_stage_numeric']  # Categorical data (e.g., 0, 1, 2)
            ancillary_data = self.one_hot_encode(ancillary_label)  # One-hot encode
            return image, ancillary_data, mask
        else:
            return image, mask
