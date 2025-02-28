import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class RoadsideCropImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, usage='train', 
                 mean=None, std=None, classwise_norm=None, 
                 use_ancillary=False, ancillary_classes=3):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths, labels, and ancillary data.
            root_dir (string): Directory with all the images and masks.
            usage (string): 'train' or 'val'. Determines if transformations are applied.
            mean (list, optional): Global mean for normalization (per band).
            std (list, optional): Global std for normalization (per band).
            classwise_norm (dict, optional): Class-wise normalization parameters 
                                             (e.g., {0: {'mean': [...], 'std': [...]}}).
            use_ancillary (bool, optional): Whether to include ancillary data or not.
            ancillary_classes (int): Number of classes in the ancillary data (for one-hot encoding).
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.usage = usage
        self.mean = mean  # Global mean if provided
        self.std = std  # Global std if provided
        self.classwise_norm = classwise_norm  # Class-wise normalization parameters
        self.use_ancillary = use_ancillary
        self.ancillary_classes = ancillary_classes

        # Check if either global or classwise normalization is provided
        if not ((self.mean and self.std) or self.classwise_norm):
            raise ValueError("Either global mean/std or class-wise normalization parameters must be provided.")

        # Build transformations (without normalization)
        self.transform = self.build_transforms()

    def build_transforms(self):
        """Build transformations for train/val usage."""
        transform_list = []

        if self.usage == 'train':
            transform_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
            ]
        elif self.usage == 'val':
            transform_list.append(transforms.Resize(size=(224, 224)))

        return transforms.Compose(transform_list)

    def one_hot_encode(self, label):
        """One-hot encode the given label."""
        return F.one_hot(torch.tensor(label), num_classes=self.ancillary_classes).float()

    def get_classwise_norm(self, label):
        """Retrieve class-specific mean and std if available."""
        if self.classwise_norm and label in self.classwise_norm:
            class_params = self.classwise_norm[label]
            mean = class_params.get('mean', [0.5] * 10)  # Default to [0.5] * 10 if not found
            std = class_params.get('std', [0.5] * 10)  # Default to [0.5] * 10 if not found
            return mean, std
        else:
            return None, None  # No class-wise normalization available

    def __len__(self):
        """Return the total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Load the image, mask, and optional ancillary data."""
        # Get paths to image and mask
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['img_chip_path'])
        mask_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['lbl_chip_path'])

        # Load the image and convert to tensor
        image = np.load(img_name)  # Load .npy image
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # [C, H, W]

        # Load the mask
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).long()

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        # Normalize the image
        if self.classwise_norm:
            # Use class-wise normalization if available
            label = self.dataframe.iloc[idx]['crop_type']
            mean, std = self.get_classwise_norm(label)
            if mean and std:
                image = transforms.functional.normalize(image, mean=mean, std=std)
            else:
                print(f"Warning: Using global normalization for label {label} as class-wise stats are missing.")
                image = transforms.functional.normalize(image, mean=self.mean, std=self.std)
        else:
            # Use global normalization
            image = transforms.functional.normalize(image, mean=self.mean, std=self.std)

        # Return data with optional ancillary information
        if self.use_ancillary:
            ancillary_data = self.one_hot_encode(self.dataframe.iloc[idx]['crop_stage_numeric'])
            return image, ancillary_data, mask
        else:
            return image, mask
