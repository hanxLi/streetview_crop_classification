import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageDraw
from chipping import generate_stacked_image
from tqdm import tqdm
from torchvision.transforms.functional import normalize


def enable_dropout(model):
    """
    Enable dropout layers during inference for MC Dropout.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def one_hot_encode(label, num_classes):
    """
    One-hot encode the given label based on the number of ancillary classes.
    """
    return F.one_hot(torch.tensor(label), num_classes=num_classes).float()

def load_ancillary_data(csv_path, image_name, num_classes):
    """
    Load and retrieve the one-hot encoded ancillary data for the given image.
    """
    df = pd.read_csv(csv_path)
    row = df[df['origin_img'] == image_name]

    if row.empty:
        raise ValueError(f"No ancillary data found for image: {image_name}")

    crop_stage_numeric = row.iloc[0]['crop_stage_numeric']
    return one_hot_encode(crop_stage_numeric, num_classes)

def predict_patch_mc_dropout(model, patch, ancillary_data, device, num_passes=10):
    """
    Predict segmentation mask and uncertainty using MC Dropout for a given patch.
    """
    # Convert patch to tensor and move to device
    patch_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
    ancillary_tensor = ancillary_data.unsqueeze(0).to(device)

    # Enable dropout layers for MC Dropout
    enable_dropout(model)

    # Collect predictions across multiple forward passes
    preds = []
    for _ in range(num_passes):
        logits = model(patch_tensor, ancillary_tensor)
        probs = F.softmax(logits, dim=1)
        _, pred = torch.max(probs, dim=1)
        preds.append(pred.cpu().numpy())

    # Stack predictions to compute mean and uncertainty
    preds = np.stack(preds)  # Shape: [num_passes, H, W]
    mean_pred = np.mean(preds, axis=0).astype(np.uint8)
    uncertainty = np.var(preds, axis=0).mean()

    return mean_pred, uncertainty

def sliding_window(image, step=16, window_size=(224, 224)):
    """
    Generate windows from the original image with overlap.
    """
    H, W, _ = image.shape
    for y in range(0, H - window_size[0] + 1, step):
        for x in range(0, W - window_size[1] + 1, step):
            yield (x, y, image[y:y + window_size[0], x:x + window_size[1]])

def stitch_predictions(image_shape, window_size, step, predictions, uncertainties):
    """
    Stitch together predictions and uncertainties from sliding window patches.
    """
    H, W = image_shape[:2]
    pred_mask = np.zeros((H, W), dtype=np.uint8)
    uncertainty_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.uint8)

    for (x, y, pred, uncertainty) in predictions:
        pred_mask[y:y + window_size[0], x:x + window_size[1]] += pred
        uncertainty_map[y:y + window_size[0], x:x + window_size[1]] += uncertainty
        count_map[y:y + window_size[0], x:x + window_size[1]] += 1

    pred_mask = np.round(pred_mask / np.maximum(count_map, 1)).astype(np.uint8)
    uncertainty_map /= np.maximum(count_map, 1)

    return pred_mask, uncertainty_map

def draw_bounding_boxes(image, mask, uncertainty_map, threshold=0.5):
    """
    Draw bounding boxes around predicted mask areas with uncertainty score.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_uncertainty = uncertainty_map[y:y+h, x:x+w].mean()
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        draw.text((x, y - 10), f"Unc: {box_uncertainty:.2f}", fill="red")

    return np.array(image_pil)

def predict_patch_mc_dropout_batch(model, patches, ancillary_data, device, num_passes=10):
    """
    Predict segmentation masks and uncertainties using MC Dropout for a batch of patches.
    """
    # Stack ancillary data to match the batch size and number of passes
    batch_size = patches.shape[0]
    patch_tensor = torch.stack([transforms.ToTensor()(patch) for patch in patches]).to(device)
    patch_tensor = patch_tensor.unsqueeze(1).repeat(1, num_passes, 1, 1, 1)  # [batch_size, num_passes, C, H, W]
    ancillary_tensor = ancillary_data.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_passes, 1).to(device)

    # Reshape to combine batch and MC passes for efficient processing
    patch_tensor = patch_tensor.view(-1, *patch_tensor.shape[2:])  # [batch_size * num_passes, C, H, W]
    ancillary_tensor = ancillary_tensor.view(-1, ancillary_tensor.shape[-1])  # [batch_size * num_passes, ancillary_dim]

    # Enable dropout layers for MC Dropout
    enable_dropout(model)

    # Perform forward passes and gather predictions
    with torch.no_grad():
        logits = model(patch_tensor, ancillary_tensor)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()  # Shape: [batch_size * num_passes, H, W]

    # Reshape predictions to separate batch and num_passes dimensions correctly
    preds = preds.reshape(batch_size, num_passes, *preds.shape[1:])  # Shape: [batch_size, num_passes, H, W]

    # Compute mean prediction and uncertainty (variance) for each patch
    mean_preds = preds.mean(axis=1).astype(np.uint8)
    uncertainties = preds.var(axis=1).mean(axis=(1, 2))  # Mean variance across spatial dimensions

    return mean_preds, uncertainties



def predict_full_image(model, image_path, csv_path, num_classes, device, step=32, window_size=(224, 224), num_passes=5, batch_size=8):
    """
    Predict the segmentation mask and uncertainty for the entire image using batched MC Dropout.
    """
    image_name = os.path.basename(image_path).split(".")[0]
    print(f"Predicting image {image_name}")
    ancillary_data = load_ancillary_data(csv_path, image_name, num_classes)

    image = cv2.imread(image_path)
    input_image = generate_stacked_image(image)
    predictions, uncertainties = [], []

    # Calculate the total number of patches for progress tracking
    H, W = input_image.shape[:2]
    num_patches = ((H - window_size[0]) // step + 1) * ((W - window_size[1]) // step + 1)

    # Initialize progress bar
    with tqdm(total=num_patches, desc="Predicting patches", unit="patch") as pbar:
        # Collect patches for batch processing
        patches = []
        positions = []
        for (x, y, patch) in sliding_window(input_image, step, window_size):
            patches.append(patch)
            positions.append((x, y))
            
            # If batch is filled, perform prediction on the batch
            if len(patches) == batch_size:
                batch_preds, batch_uncertainties = predict_patch_mc_dropout_batch(
                    model, np.array(patches), ancillary_data, device, num_passes=num_passes
                )
                predictions.extend((x, y, pred, unc) for (x, y), pred, unc in zip(positions, batch_preds, batch_uncertainties))
                patches.clear()
                positions.clear()
                pbar.update(batch_size)  # Update progress bar by the batch size

        # Process any remaining patches
        if patches:
            batch_preds, batch_uncertainties = predict_patch_mc_dropout_batch(
                model, np.array(patches), ancillary_data, device, num_passes=num_passes
            )
            predictions.extend((x, y, pred, unc) for (x, y), pred, unc in zip(positions, batch_preds, batch_uncertainties))
            pbar.update(len(patches))  # Update progress bar by the remaining patches

    # Stitch the predictions back together
    pred_mask, uncertainty_map = stitch_predictions(input_image.shape, window_size, step, predictions, uncertainties)

    # Visualize results
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    result_image = draw_bounding_boxes(original_image_rgb, pred_mask, uncertainty_map)

    Image.fromarray(result_image).show()

    return pred_mask, uncertainty_map

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


# def simple_predict_full_image(model, image_path, csv_path, num_classes, device, step=32, window_size=(224, 224)):
#     """
#     Perform a simple sliding window prediction on the entire image, without uncertainty estimation.
    
#     Args:
#         model (torch.nn.Module): The model for prediction.
#         image_path (str): Path to the input image.
#         csv_path (str): Path to the ancillary data CSV.
#         num_classes (int): Number of classes for segmentation.
#         device (torch.device): Device to run the prediction on.
#         step (int): Step size for the sliding window.
#         window_size (tuple): Size of each sliding window patch.
    
#     Returns:
#         np.array: Predicted mask for the entire image.
#     """
#     image_name = Path(image_path).stem
#     print(f"Predicting image {image_name} without uncertainty")
#     ancillary_data = load_ancillary_data(csv_path, image_name, num_classes)

#     # Load and prepare the input image
#     image = cv2.imread(image_path)
#     input_image = generate_stacked_image(image)
#     H, W = input_image.shape[:2]

#     # Initialize output mask for the entire image as int64 for accumulation
#     pred_mask = np.zeros((H, W), dtype=np.int64)
#     count_map = np.zeros((H, W), dtype=np.uint8)

#     # Sliding window prediction
#     for y in range(0, H - window_size[0] + 1, step):
#         for x in range(0, W - window_size[1] + 1, step):
#             # Extract the window
#             window = input_image[y:y + window_size[0], x:x + window_size[1]]
            
#             # Convert the window to tensor format and move to the device
#             window_tensor = transforms.ToTensor()(window).unsqueeze(0).to(device)
#             ancillary_tensor = ancillary_data.unsqueeze(0).to(device)

#             # Perform a single forward pass with no dropout or uncertainty estimation
#             with torch.no_grad():
#                 logits = model(window_tensor, ancillary_tensor)
#                 probs = F.softmax(logits, dim=1)
#                 _, pred = torch.max(probs, dim=1)

#             # Add the predicted patch to the output mask
#             pred = pred.squeeze(0).cpu().numpy()  # Shape: (224, 224)
#             pred_mask[y:y + window_size[0], x:x + window_size[1]] += pred
#             count_map[y:y + window_size[0], x:x + window_size[1]] += 1

#     # Normalize the mask by dividing by the count map and convert to uint8
#     pred_mask = np.round(pred_mask / np.maximum(count_map, 1)).astype(np.uint8)

#     return pred_mask


def simple_predict_full_image(model, image_path, csv_path, num_classes, device, step=32, window_size=(224, 224)):
    """
    Perform a sliding window prediction on the entire image using argmax for final mask aggregation.
    
    Args:
        model (torch.nn.Module): The model for prediction.
        image_path (str): Path to the input image.
        csv_path (str): Path to the ancillary data CSV.
        num_classes (int): Number of classes for segmentation.
        device (torch.device): Device to run the prediction on.
        step (int): Step size for the sliding window.
        window_size (tuple): Size of each sliding window patch.
    
    Returns:
        np.array: Predicted mask for the entire image.
    """
    image_name = Path(image_path).stem
    print(f"Predicting image {image_name} without uncertainty")
    ancillary_data = load_ancillary_data(csv_path, image_name, num_classes)

    # Load and prepare the input image
    image = cv2.imread(image_path)
    input_image = generate_stacked_image(image)
    H, W = input_image.shape[:2]

    # Initialize score map for each class and a count map for normalization
    class_score_map = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.uint8)

    # Sliding window prediction
    for y in range(0, H - window_size[0] + 1, step):
        for x in range(0, W - window_size[1] + 1, step):
            # Extract the window and resize to match training context (512x512)
            window = input_image[y:y + window_size[0], x:x + window_size[1]]
            # window_resized = cv2.resize(window, (512, 512))

            # Convert the resized window to tensor format and move to device
            # window_tensor = transforms.ToTensor()(window_resized).unsqueeze(0).to(device)
            window_tensor = transforms.ToTensor()(window).unsqueeze(0).to(device)

            ancillary_tensor = ancillary_data.unsqueeze(0).to(device)

            # Perform a single forward pass with no dropout or uncertainty estimation
            with torch.no_grad():
                logits = model(window_tensor, ancillary_tensor)
                probs = F.softmax(logits, dim=1)  # Shape: (1, num_classes, 224, 224)
                probs_resized = F.interpolate(probs, size=window_size, mode='nearest')  # Resize back to 224x224

            # Convert probabilities to numpy and accumulate in the score map
            probs_np = probs_resized.squeeze(0).cpu().numpy()  # Shape: (num_classes, 224, 224)
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