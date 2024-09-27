import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import geopandas as gpd
import pandas as pd
import cv2
import time
import random

from utils import *

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def filter_contours(mask, min_area=500):
    """
    Function to filter out small contours from the mask.
    Only keeps contours larger than 'min_area'.
    """
    # Find all contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask to draw filtered contours
    filtered_mask = np.zeros_like(mask)

    # Draw contours that are larger than the minimum area
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_mask

def get_mask_prompt(img, pixels_below=[100, 300], num_points=3):
    """
    Function to randomly select points below the skyline in a sky mask and ensure they are below
    the average y-value of the skyline. If no valid pixels are found, return the center pixel.
    
    :param img: Input image.
    :param pixels_below: Number of pixels below the skyline to choose the points.
    :param num_points: Number of random points to select.
    :return: List of (x, y) points below the skyline, or a single center pixel if no valid points are found.
    """
    # Convert image to HSV and create a binary mask for the sky
    im_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, _, _ = cv2.split(im_hsv)
    mask = cv2.inRange(h, 90, 255)  # Adjust the hue range for your specific sky color
    mask = filter_contours(mask)  # Filter small areas

    height, width = mask.shape
    
    # Initialize the skyline (bottom-most sky pixel in each column)
    skyline = np.full(width, -1)  # Initialize to -1 to indicate no sky found
    for col in range(width):
        for row in range(height - 1, -1, -1):  # Loop from bottom to top
            if mask[row, col] == 255:  # Find bottom-most sky pixel
                skyline[col] = row
                break  # Stop after finding the first sky pixel from the bottom

    # Calculate the average y-value of the skyline
    valid_skyline_pixels = skyline[skyline != -1]  # Exclude columns where no sky was found
    if len(valid_skyline_pixels) == 0:
        # Return the center pixel if no valid points are found
        return [[width // 2, height // 2]]
    
    avg_skyline_y = np.mean(valid_skyline_pixels)  # Calculate the average y-value of the skyline
    
    # Eliminate skyline pixels that are more than 150 pixels off the average
    for col in range(width):
        if skyline[col] != -1 and abs(skyline[col] - avg_skyline_y) > 150:
            skyline[col] = -1  # Invalidate the pixel by setting it to -1

    # Collect valid skyline points (columns with valid bottom-most sky pixels)
    valid_cols = [col for col in range(width) if skyline[col] != -1]

    # If fewer than 3 valid points are found, return the center point
    if len(valid_cols) < num_points:
        return [[width // 2, height // 2]]
    
    # Randomly select the required number of columns
    selected_cols = random.sample(valid_cols, num_points)

    points_below = []

    for col in selected_cols:
        # Shift the point randomly below the skyline
        point_below = skyline[col] + random.randint(pixels_below[0], pixels_below[1])
        # Ensure the point is within bounds of the image
        if point_below >= height:
            point_below = height - 1  # Adjust to stay within bounds

        points_below.append([col, point_below])

    return points_below


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 100/255, 100/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_best_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, title=None):
    """
    Function to display only the mask with the highest score.
    
    Parameters:
    - image: The input image to show the mask on.
    - masks: A list of masks.
    - scores: A list of scores for each mask.
    - point_coords: Optional, coordinates for points.
    - box_coords: Optional, coordinates for boxes.
    - input_labels: Optional, labels for points.
    - borders: Boolean to indicate if borders around masks should be shown.
    """
    # Find the index of the mask with the highest score
    max_index = np.argmax(scores)
    max_mask = masks[max_index]
    max_score = scores[max_index]
    
    # Display the image with the mask that has the highest score

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))


    ax[0].imshow(image)
    ax[0].set_title(f"Original Image")
    ax[0].axis('off')

    ax[1].imshow(image)
    ax[1].axis("off")
    ax[1].set_title(f"Mask with Highest Score: {max_score:.3f}")
    
    # Show the mask with the highest score
    show_mask(max_mask, ax[1], borders=borders)
    
    # Optionally display points and boxes if they are provided
    if point_coords is not None:
        assert input_labels is not None, "input_labels must be provided with point_coords"
        show_points(point_coords, input_labels, ax[1])
    if box_coords is not None:
        show_box(box_coords, ax[1])
    
    # Show the title with the highest score
    if title:
        fig.suptitle(title, fontsize=18, y=0.8)
    fig.subplots_adjust(top=0.85, wspace=0.05)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

def sam2_mask_generate(df_path, idx, model_params, point = None):

    df = gpd.read_file(df_path, driver="GEOJSON")

    image = Image.open(df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/"))
    image = np.array(image.convert("RGB"))
    time = df.loc[idx, "Timestamp"]

    title_text = f"{df.loc[idx, 'crop_type']} | {df.loc[idx, 'Name']} | {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"

    h, v, _ = image.shape

    chkpt, cfg_path, device = model_params
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    sam2_model = build_sam2(cfg_path, chkpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    if not point:
        point = get_mask_prompt(image)

    lbl_list = []
    for i in range(len(point)):
        lbl_list.append(1)
    point.append([v // 2, h - 300])
    lbl_list.append(0)
    
    input_point = np.array(point)
    input_label = np.array(lbl_list)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True)

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    show_best_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True, title=title_text)