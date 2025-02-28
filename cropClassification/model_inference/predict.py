import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
from cropClassification.utils.chipping import generate_stacked_image
from tqdm import tqdm

def one_hot_encode(label, num_classes):
    """
    One-hot encode the given label based on the number of ancillary classes.
    """
    return F.one_hot(torch.tensor(label), num_classes=num_classes).float()

def load_ancillary_data_from_summary(summary_df, image_name, num_classes):
    """
    Load and retrieve the one-hot encoded ancillary data for the given image from the summary DataFrame.
    
    Args:
        summary_df (pd.DataFrame): DataFrame containing image metadata
        image_name (str): Name of the image (without extension)
        num_classes (int): Number of classes for one-hot encoding
    
    Returns:
        torch.Tensor: One-hot encoded crop stage numeric value
    """
    # Find the row for this image
    row = summary_df[summary_df['image_name'] == image_name]

    if row.empty:
        raise ValueError(f"No ancillary data found for image: {image_name}")

    # Extract the crop stage information - map stage to numeric value
    crop_stage = row.iloc[0]['crop_stage']
    
    # Map the crop stage to numeric value
    stage_to_numeric = {'planting': 0, 'growing': 1, 'harvesting': 2}
    
    if crop_stage in stage_to_numeric:
        crop_stage_numeric = stage_to_numeric[crop_stage]
    else:
        # Default to 0 if unknown stage
        print(f"Warning: Unknown crop stage '{crop_stage}' for image {image_name}. Using default value 0.")
        crop_stage_numeric = 0
        
    return one_hot_encode(crop_stage_numeric, num_classes)

def simple_predict_full_image(model, image_path, summary_df, num_classes, device, norm_params, step=32, window_size=(224, 224)):
    """
    Perform a sliding window prediction on the entire image using argmax for final mask aggregation.
    
    Args:
        model (torch.nn.Module): The model for prediction.
        image_path (str): Path to the input image.
        summary_df (pd.DataFrame): DataFrame containing image metadata.
        num_classes (int): Number of classes for segmentation.
        device (torch.device): Device to run the prediction on.
        norm_params (tuple): Mean and standard deviation for normalization.
        step (int): Step size for the sliding window.
        window_size (tuple): Size of each sliding window patch.
    
    Returns:
        np.array: Predicted mask for the entire image.
    """
    model.eval()
    image_name = Path(image_path).stem
    print(f"Predicting image {image_name}")
    
    # Get ancillary data from the summary DataFrame
    ancillary_data = load_ancillary_data_from_summary(summary_df, image_name, num_classes)

    # Load and prepare the input image
    image = cv2.imread(image_path)  # (H, W, 3) if using RGB
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    input_image = generate_stacked_image(image)  # Ensure it outputs (H, W, C) with C=10

    H, W, C = input_image.shape  # Check number of bands

    # Convert NumPy (H, W, C) → Tensor (C, H, W) and normalize
    input_image_tensor = torch.from_numpy(input_image).permute(2, 0, 1).float()
    mean, std = norm_params
    input_image_tensor = transforms.functional.normalize(input_image_tensor, mean=mean, std=std)

    # Initialize score map for each class and a count map for normalization
    class_score_map = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # Sliding window prediction
    for y in range(0, H - window_size[0] + 1, step):
        for x in range(0, W - window_size[1] + 1, step):
            # Extract the window (C, H, W)
            window_tensor = input_image_tensor[:, y:y + window_size[0], x:x + window_size[1]].unsqueeze(0).to(device)

            ancillary_tensor = ancillary_data.unsqueeze(0).to(device)

            # Perform a single forward pass with no dropout or uncertainty estimation
            with torch.no_grad():
                logits = model(window_tensor, ancillary_tensor)
                probs = F.softmax(logits, dim=1)  # Shape: (1, num_classes, 224, 224)

            # Convert probabilities to numpy and accumulate in the score map
            probs_np = probs.squeeze(0).cpu().numpy()  # Shape: (num_classes, 224, 224)
            for c in range(num_classes):
                class_score_map[c, y:y + window_size[0], x:x + window_size[1]] += probs_np[c]
            count_map[y:y + window_size[0], x:x + window_size[1]] += 1

    # Normalize by dividing by the count map where count_map > 0
    count_map = np.maximum(count_map, 1)  # Avoid division by zero
    for c in range(num_classes):
        class_score_map[c] /= count_map

    # Take argmax along the class dimension to get final class predictions
    pred_mask = np.argmax(class_score_map, axis=0).astype(np.uint8)

    return pred_mask

def overlay_prediction(image_path, pred_mask, alpha=0.6, colors=None):
    """
    Overlay the predicted mask on the original image with distinct colors for each class.

    Args:
        image_path (str): Path to the original image file.
        pred_mask (np.array): Predicted mask to overlay.
        alpha (float): Transparency level for overlay.
        colors (list or None): List of RGB tuples representing colors for each class (0, 1, 2, ...).

    Returns:
        np.array: Combined image with the mask overlay.
    """
    # Load the original image and convert to RGB
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Define default colors if none are provided
    if colors is None:
        colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # Default: Black, Red, Green

    # Create a blank RGB mask and fill with class colors
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        color_mask[pred_mask == class_id] = color

    # Blend the original image and the color mask
    overlay = cv2.addWeighted(original_image_rgb, 1 - alpha, color_mask, alpha, 0)
    
    return overlay

def save_prediction_results(image_path, pred_mask, overlay_image=None, save_dir="predictions"):
    """
    Save the prediction mask and overlay image.

    Args:
        image_path (str): Path to the input image.
        pred_mask (np.array): Predicted mask for the entire image.
        overlay_image (np.array or None): Overlay image to save.
        save_dir (str): Directory to save the results.
    """
    # Ensure the save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Extract the base name of the input image (e.g., "image.jpg" -> "image")
    image_name = Path(image_path).stem
    
    # Save the predicted mask
    mask_save_path = Path(save_dir) / f"{image_name}_pred_mask.png"
    Image.fromarray(pred_mask.astype(np.uint8)).save(mask_save_path)
    print(f"Prediction mask saved to: {mask_save_path}")

    # Save the overlay image if provided
    if overlay_image is not None:
        overlay_save_path = Path(save_dir) / f"{image_name}_overlay.png"
        Image.fromarray(overlay_image).save(overlay_save_path)
        print(f"Overlay image saved to: {overlay_save_path}")