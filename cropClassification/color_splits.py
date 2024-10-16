import cv2
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np

from utils import *

def load_and_convert_image(df, idx, color_space):
    """
    Loads an image from the specified path and converts it to the given color space.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata (e.g., save paths, timestamps).
    idx : int
        Index of the image in the DataFrame to process.
    color_space : str, optional
        The color space to convert the image to ('LAB', 'HSV', 'HLS').
        
    Returns
    -------
    im_rgb : np.ndarray
        The image in RGB format.
    converted_image : np.ndarray
        The image converted to the requested color space.
    """
    path = df.loc[idx, "save_path"].replace("/home/hanxli/", "/workspace/")
    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    
    if color_space == 'LAB':
        converted_image = cv2.cvtColor(imgdata, cv2.COLOR_BGR2LAB)
    elif color_space == 'HSV':
        converted_image = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HSV)
    elif color_space == 'HLS':
        converted_image = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HLS)
    else:
        raise ValueError("Unsupported color space")
    
    return im_rgb, converted_image

def generate_title(df, idx):
    """
    Generates a plot title based on the crop type, name, and timestamp from the DataFrame.
    """
    _time = df.loc[idx, "Timestamp"]
    title_text = f"{df.loc[idx, 'crop_type']} | {df.loc[idx, 'Name']} | "\
                 f"{_time.year}/{_time.month}/{_time.day} | "\
                 f"{check_crop_stage(df.loc[idx, 'crop_type'], _time.month)} stage"
    return title_text

def plot_images_with_channels(im_rgb, channels, channel_names, title_text):
    """
    Plots the RGB image and the channels from a converted image in a 2x2 grid.

    Parameters
    ----------
    im_rgb : np.ndarray
        The RGB image to display.
    channels : list of np.ndarray
        The list of image channels (e.g., [l, a, b]).
    channel_names : list of str
        The names of the channels for plotting (e.g., ['L', 'A', 'B']).
    title_text : str
        The main title for the plot.
    """
    # Create a 2x2 subplot grid
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    # Plot the RGB image in the first subplot
    ax[0, 0].imshow(im_rgb)
    ax[0, 0].set_title("RGB Image")
    ax[0, 0].axis('off')

    # Plot the channels in the remaining subplots
    for i, (channel, name) in enumerate(zip(channels, channel_names)):
        row, col = divmod(i + 1, 2)  # Adjust indexing to start plotting at (0, 1)
        ax[row, col].imshow(channel, cmap='gray')
        ax[row, col].set_title(f"{name} Channel [{np.min(channel)} - {np.max(channel)}]")
        ax[row, col].axis('off')

    # If fewer than 4 channels, turn off the remaining subplot(s)
    if len(channels) < 3:
        for i in range(len(channels) + 1, 4):
            row, col = divmod(i, 2)
            ax[row, col].axis('off')

    # Set the main title
    fig.suptitle(title_text, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def mask_with_lab(df, idx, a_thresh=120, b_thresh=140, soybean=False, return_mask=True):
    """
    Applies a mask based on thresholds in the LAB color space.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata.
    idx : int
        Index of the image in the DataFrame.
    a_thresh : int, optional
        Threshold for the 'a' channel in LAB (default is 120).
    b_thresh : int, optional
        Threshold for the 'b' channel in LAB (default is 140).
    soybean : bool, optional
        If True, applies a specific logic for soybean masking (default is False).
    """
    # Load the image and convert to LAB
    im_rgb, im_lab = load_and_convert_image(df, idx, color_space='LAB')

    # Split LAB channels
    _, a, b = cv2.split(im_lab)

    # Apply thresholds
    thresh_a = cv2.threshold(a, a_thresh, 255, cv2.THRESH_BINARY_INV)[1]
    thresh_b = cv2.threshold(b, b_thresh, 255, cv2.THRESH_BINARY_INV)[1]

    if soybean:
        thresh_b = np.logical_not(thresh_b).astype(np.uint8)
        mask = np.logical_and(thresh_a, thresh_b).astype(np.uint8)
    else:
        mask = (~(thresh_a == 0) | ~(thresh_b == 0)).astype(np.uint8)
        mask = np.logical_not(mask).astype(np.uint8)

    # Generate title and plot the images
    title_text = generate_title(df, idx)
    plot_images_with_channels(im_rgb, [mask], ['Crop Mask from LAB Transform'], title_text)
    if return_mask:
        return mask

def mask_with_hsv(df, idx, h_thresh_range=None, s_thresh_range=None, return_mask=True):
    """
    Applies a mask based on thresholds in the HSV color space.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata.
    idx : int
        Index of the image in the DataFrame.
    h_thresh_range : list, optional
        Range of thresholds for the 'h' channel in HSV (default is [40, 90]).
    s_thresh_range : list, optional
        Range of thresholds for the 's' channel in HSV (default is [120, 255]).
    """

    if h_thresh_range is None:
        h_thresh_range=[40, 90]
    if s_thresh_range is None:
        s_thresh_range=[120, 255]
    # Load the image and convert to HSV
    im_rgb, im_hsv = load_and_convert_image(df, idx, color_space='HSV')

    # Split HSV channels
    h, s, _ = cv2.split(im_hsv)

    # Apply thresholds
    range_h = cv2.inRange(h, h_thresh_range[0], h_thresh_range[1])
    range_s = cv2.inRange(s, s_thresh_range[0], s_thresh_range[1])
    mask = np.logical_and(range_h, range_s).astype(np.uint8)

    # Generate title and plot the images
    title_text = generate_title(df, idx)
    plot_images_with_channels(im_rgb, [mask], ['Crop Mask from HSV Transform'], title_text)
    if return_mask:
        return mask

def hls_split(df, idx):
    """
    Splits the image into HLS channels and displays them.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata.
    idx : int
        Index of the image in the DataFrame.
    
    Returns
    -------
    h : np.ndarray
        Hue channel of the image.
    l : np.ndarray
        Lightness channel of the image.
    s : np.ndarray
        Saturation channel of the image.
    """
    # Load the image and convert to HLS
    im_rgb, im_hls = load_and_convert_image(df, idx, color_space='HLS')

    # Split HLS channels
    h, l, s = cv2.split(im_hls)

    # Generate title and plot the images
    title_text = generate_title(df, idx)
    plot_images_with_channels(im_rgb, [h, l, s], ['H', 'L', 'S'], title_text)

    return h, l, s

def hsv_split(df, idx):
    """
    Splits the image into HSV channels and displays them.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata.
    idx : int
        Index of the image in the DataFrame.
    
    Returns
    -------
    h : np.ndarray
        Hue channel of the image.
    s : np.ndarray
        Saturation channel of the image.
    v : np.ndarray
        Value channel of the image.
    """
    # Load the image and convert to HSV
    im_rgb, im_hsv = load_and_convert_image(df, idx, color_space='HSV')

    # Split HSV channels
    h, s, v = cv2.split(im_hsv)

    # Generate title and plot the images
    title_text = generate_title(df, idx)
    plot_images_with_channels(im_rgb, [h, s, v], ['H', 'S', 'V'], title_text)

    return h, s, v

def lab_split(df, idx):
    """
    Splits the image into LAB channels and displays them.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata (e.g., save paths, timestamps).
    idx : int
        Index of the image in the DataFrame to process.
    
    Returns
    -------
    l : np.ndarray
        Lightness channel of the image.
    a : np.ndarray
        'a' channel of the image.
    b : np.ndarray
        'b' channel of the image.
    """
    # Load the image and convert it to LAB
    im_rgb, im_lab = load_and_convert_image(df, idx, color_space='LAB')

    # Split LAB channels
    l, a, b = cv2.split(im_lab)

    # Generate the title
    title_text = generate_title(df, idx)

    # Plot the original image and the LAB channels
    plot_images_with_channels(im_rgb, [l, a, b], ['L', 'A', 'B'], title_text)

    return l, a, b

def normalized_grayscale_convert(df, idx, threshold=None):
    """
    Convert an image to normalized grayscale reflectance and optionally 
    threshold the reflectance values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata (e.g., save paths, timestamps).
    idx : int
        Index of the image in the DataFrame to process.
    threshold : list or tuple, optional
        A list or tuple containing two values: (min_threshold, max_threshold).
        If None, a default threshold [0, 0.2] is applied.

    Returns
    -------
    reflectance : np.ndarray
        The normalized grayscale reflectance of the image.
    thresholded_image : np.ndarray
        The binary image after applying the threshold.
    """

    # Set default threshold if none is provided
    if threshold is None:
        threshold = [0, 0.2]

    # Validate the threshold input
    if not isinstance(threshold, (list, tuple)) or len(threshold) != 2:
        raise ValueError("Threshold must be a list or tuple with two values: (min_threshold, max_threshold).")

    min_threshold, max_threshold = threshold

    # Load image and extract metadata for title
    path = df.loc[idx, "save_path"].replace("/home/hanxli/", "/workspace/")
    _time = df.loc[idx, "Timestamp"]
    title_text = f"{df.loc[idx, 'crop_type']} | {df.loc[idx, 'Name']} | "\
                 f"{_time.year}/{_time.month}/{_time.day} | "\
                 f"{check_crop_stage(df.loc[idx, 'crop_type'], _time.month)} stage"

    # Read the image and convert to RGB
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the R, G, and B channels to the range [0, 1]
    image_normalized = image_rgb / 255.0

    # Calculate the reflectance (average across the R, G, B channels)
    reflectance = np.mean(image_normalized, axis=2)

    # Apply the threshold to create a binary mask
    thresholded_image = (reflectance >= min_threshold) & (reflectance <= max_threshold)
    thresholded_image = thresholded_image.astype(np.uint8)  # Convert boolean mask to binary (0s and 1s)

    # Plot the original RGB image and thresholded reflectance image
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Display the RGB image
    ax[0].imshow(image_rgb)
    ax[0].set_title("RGB Image")
    ax[0].axis("off")

    # Display the thresholded reflectance image
    ax[1].imshow(thresholded_image, cmap='gray')
    ax[1].set_title("Reflectance Thresholded Image")
    ax[1].axis("off")

    # Add a title and adjust layout
    fig.suptitle(title_text, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    return reflectance, thresholded_image
