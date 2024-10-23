import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from torchvision import transforms
from PIL import Image, ImageDraw

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

def preprocess_image(image_path):
    """
    Load and preprocess the original image to produce RGB, HSV, and LAB input.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV and LAB color spaces
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # Concatenate all channels (RGB + HSV + LAB)
    return np.concatenate([image_rgb, image_hsv, image_lab], axis=-1)

def predict_patch(model, patch, ancillary_data, device):
    """
    Predict the segmentation mask and uncertainty for a given patch.
    """
    patch_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
    ancillary_tensor = ancillary_data.unsqueeze(0).to(device)

    logits, log_var = model(patch_tensor, ancillary_tensor)

    probs = F.softmax(logits, dim=1)
    _, pred = torch.max(probs, dim=1)
    uncertainty = torch.exp(log_var).mean().item()

    return pred.squeeze(0).cpu().numpy(), uncertainty

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

def predict_full_image(model, image_path, csv_path, num_classes, device, step=16, window_size=(224, 224)):
    """
    Predict the segmentation mask and uncertainty for the entire image.
    """
    image_name = os.path.basename(image_path)
    ancillary_data = load_ancillary_data(csv_path, image_name, num_classes)

    input_image = preprocess_image(image_path)
    predictions, uncertainties = [], []

    for (x, y, patch) in sliding_window(input_image, step, window_size):
        pred, uncertainty = predict_patch(model, patch, ancillary_data, device)
        predictions.append((x, y, pred, uncertainty))
        uncertainties.append(uncertainty)

    pred_mask, uncertainty_map = stitch_predictions(input_image.shape, window_size, step, predictions, uncertainties)

    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    result_image = draw_bounding_boxes(original_image_rgb, pred_mask, uncertainty_map)

    Image.fromarray(result_image).show()

    return pred_mask, uncertainty_map
