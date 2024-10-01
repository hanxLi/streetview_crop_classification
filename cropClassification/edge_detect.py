import os
import fiona
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import time
from PIL import Image

from utils import *

def canny_edge(df, df_index, threshold=None, enhance=False, return_mask=True):
    """
    Apply Canny edge detection to an image specified in the DataFrame and display the results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing metadata and paths to images.
    df_index : int
        Index of the image in the DataFrame to process.
    threshold : tuple, optional
        A tuple (lower_threshold, upper_threshold) for the Canny edge detection.
        If None, Otsu's method is used to determine thresholds.
    enhance : bool, optional
        Whether to enhance the image by increasing its contrast (default is False).

    Returns
    -------
    edges : np.ndarray
        Edge-detected image using the Canny method.
    """

    # Extract metadata for title
    _time = df.loc[df_index, "Timestamp"]
    title_text = f"{df.loc[df_index, 'crop_type']} | {df.loc[df_index, 'Name']} |"\
                 f"{_time.year}/{_time.month}/{_time.day} | "\
                 f"{check_crop_stage(df.loc[df_index, 'crop_type'], _time.month)} stage"

    # Load the image
    img_path = df.loc[df_index, "save_path"].replace("/home/hanxli/data/",
                                                      "/Users/steeeve/Documents/csiss/")
    imgdata = cv2.imread(img_path)

    # Handle case if image is not found
    if imgdata is None:
        raise FileNotFoundError(f"Image at {img_path} could not be loaded.")

    # Convert to grayscale
    im_gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)

    # Copy the grayscale image and optionally enhance it
    img_copy = im_gray.copy()
    if enhance:
        img_copy = np.clip(img_copy * 3, 0, 255).astype(np.uint8)  # Clipping to avoid overflow

    # Apply Gaussian blur to smooth the image
    img_blur = cv2.GaussianBlur(img_copy, (5, 5), 0)

    # Determine thresholds for Canny edge detection
    if threshold:
        if not isinstance(threshold, tuple) or len(threshold) != 2 or threshold[0] < 0 or threshold[1] < 0:
            raise ValueError("Threshold must be a tuple of two non-negative values (lower_threshold, upper_threshold).")
        lower_threshold, upper_threshold = threshold
    else:
        # Use Otsu's method if no threshold is provided
        otsu_thresh, _ = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_threshold = otsu_thresh * 0.5
        upper_threshold = otsu_thresh * 1.5

    # Apply Canny edge detection
    edges = cv2.Canny(img_blur, int(lower_threshold), int(upper_threshold))

    # Plot grayscale and edge-detected images side by side
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].imshow(im_gray, cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Gray Scale")

    ax[1].imshow(edges, cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("Edge")

    # Set the title and adjust the layout
    fig.suptitle(title_text, fontsize=16)
    plt.tight_layout()
    plt.show()
    if return_mask:
        return edges

def calc_edge_pair(img, threshold=None, enhance=False, return_mask=True):
    """
    Calculate the edges of an image using the Canny edge detector and optional enhancement.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale).
    threshold : tuple, optional
        A tuple containing the lower and upper thresholds for Canny edge detection (default is None).
        If None, Otsu's method is used to calculate the thresholds.
    enhance : bool, optional
        If True, enhances the image by increasing contrast (default is False).

    Returns
    -------
    edges : np.ndarray
        The edges detected in the image.
    """

    # Ensure the input image is a copy to avoid modifying the original
    img_copy = img.copy()

    # Apply enhancement if needed (avoiding values above 255)
    if enhance:
        img_copy = np.clip(img_copy * 3, 0, 255).astype(np.uint8)  # Clipping to avoid overflow

    # Apply Gaussian blur to smooth the image
    img_blur = cv2.GaussianBlur(img_copy, (5, 5), 0)

    # Determine thresholds for Canny edge detection
    if threshold:
        if not isinstance(threshold, tuple) or len(threshold) != 2 or threshold[0] < 0 or threshold[1] < 0:
            raise ValueError("Threshold must be a tuple of two non-negative values (lower, upper).")
        lower_threshold, upper_threshold = threshold
    else:
        # Use Otsu's method if no threshold is provided
        otsu_thresh, _ = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_threshold = otsu_thresh * 0.5
        upper_threshold = otsu_thresh * 1.5

    # Perform Canny edge detection
    edges = cv2.Canny(img_blur, int(lower_threshold), int(upper_threshold))

    # Plot the original (or enhanced) and edge-detected images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img_copy, cmap='gray')
    ax[0].set_title('Original/Enhanced Image')
    ax[0].axis('off')

    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title(f'Edges [{np.min(edges)} - {np.max(edges)}]')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

    if return_mask:
        return edges

