import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from utils import *
from color_splits import *

#################
# Function to compute Gabor Feature Extraction
def build_gabor_kernels():
    """
    Build a set of Gabor kernels with various orientations, wavelengths, and other parameters.

    Returns
    -------
    kernels : list
        A list of Gabor kernels (filters) with different parameters.
    """
    kernels = []
    ksize = 31  # Size of the filter (width and height)

    # Loop over a set of theta (orientation), sigma (variance),
    # lambda (wavelength), and gamma (aspect ratio)
    for theta in np.arange(0, np.pi, np.pi / 8):  # Range of orientations
        for sigma in (3, 5, 7):  # Variance (sigma) of the Gaussian envelope
            for lambd in np.arange(10, 30, 5):  # Wavelength of the sinusoidal factor
                for gamma in (0.5, 1.0):  # Aspect ratio
                    kernel = cv2.getGaborKernel((ksize, ksize), 
                                                sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
    return kernels

def apply_gabor_filters(image, kernels):
    """
    Apply the Gabor filters to the input image and return the filtered images.

    Parameters
    ----------
    image : np.ndarray
        The input image (grayscale or color).
    kernels : list
        A list of Gabor kernels (filters).

    Returns
    -------
    filtered_images : list
        A list of images, each of which is the result of applying one of the 
        Gabor kernels to the input image.
    """
    filtered_images = []

    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:  # Check if it's a color image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply each Gabor filter to the image
    for kernel in kernels:
        filtered_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered_img)

    return filtered_images

def extract_gabor_features(filtered_images):
    """
    Extract Gabor features by flattening the filtered images and concatenating the feature vectors.

    Parameters
    ----------
    filtered_images : list
        A list of filtered images obtained by applying Gabor filters.

    Returns
    -------
    combined_feature : np.ndarray
        A 1D feature vector combining features from all the filtered images.
    """
    features = []

    # Flatten each filtered image into a 1D feature vector
    for filtered_img in filtered_images:
        features.append(filtered_img.flatten())

    # Concatenate all the features into one large vector
    combined_feature = np.concatenate(features)

    return combined_feature

def compute_similarity(features_img1, features_img2, metric='euclidean'):
    """
    Compute the similarity between two feature vectors using the specified metric.

    Parameters
    ----------
    features_img1 : np.ndarray
        Feature vector for the first image.
    features_img2 : np.ndarray
        Feature vector for the second image.
    metric : str, optional
        The metric to use for computing similarity. Options are 'euclidean' or 'cosine'.
        Default is 'euclidean'.

    Returns
    -------
    float
        The similarity score. For Euclidean distance, smaller values indicate higher similarity.
        For Cosine similarity, a value closer to 1 indicates higher similarity.
    
    Raises
    ------
    ValueError
        If an unsupported metric is provided or if the input feature vectors 
        are of different lengths.
    """
    # Ensure both feature vectors have the same length
    if len(features_img1) != len(features_img2):
        raise ValueError("Feature vectors must have the same length.")

    if metric == 'euclidean':
        # Compute Euclidean distance (smaller means more similar)
        return distance.euclidean(features_img1, features_img2)

    elif metric == 'cosine':
        # Compute Cosine similarity (1 means identical, 0 means completely different)
        return cosine_similarity([features_img1], [features_img2])[0][0]

    else:
        raise ValueError(f"Unsupported metric: {metric}." \
                         "Supported metrics are 'euclidean' and 'cosine'.")

def check_pattern_match_gabor(df, idx, exp_img, kernel_size=300, overlap=None):
    """
    Check the most similar region in an image using Gabor filters and cosine similarity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing metadata and image paths.
    idx : int
        Index of the image in the DataFrame to be checked.
    exp_img : np.ndarray
        The experimental or reference image to compare with.
    kernel_size : int, optional
        The size of the sliding window to extract patches (default is 300).
    overlap : int, optional
        Stride or overlap between patches. If None, patches are non-overlapping.

    Returns
    -------
    largest_cos : list
        The three most similar patches based on cosine similarity.
    """
    # Load the image from the DataFrame
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    imgdata = cv2.imread(path)

    # Handle image loading errors
    if imgdata is None:
        raise FileNotFoundError(f"Image at path {path} could not be loaded.")

    # Convert image to RGB and grayscale
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)

    # Image dimensions
    h, w = im_gray.shape

    # Ensure kernel size is valid
    if kernel_size > h or kernel_size > w:
        raise ValueError(f"Kernel size {kernel_size} is larger than the image dimensions.")

    # Create a figure for visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(im_rgb)
    ax[0].axis("off")

    # Create a Gabor mask for highlighting the most similar regions
    gabor_out = np.full_like(im_gray, 255)

    # List to store cosine similarities and corresponding coordinates
    gabor_list = []
    coords = []

    # Build Gabor kernels and filter the experimental image
    kernels = build_gabor_kernels()
    exp_resized = cv2.resize(exp_img, (kernel_size, kernel_size))
    exp_filtered = apply_gabor_filters(exp_resized, kernels)
    feature_img_exp = extract_gabor_features(exp_filtered)

    # Set up the stride based on overlap
    stride = overlap if overlap else kernel_size

    # Loop through the image with a sliding window
    for y in range(0, h - kernel_size + 1, stride):
        for x in range(0, w - kernel_size + 1, stride):
            # Extract the patch using the current window position
            patch = im_gray[y:y + kernel_size, x:x + kernel_size]
            target_filtered = apply_gabor_filters(patch, kernels)
            feature_img_tar = extract_gabor_features(target_filtered)

            # Compute cosine similarity between the reference image and patch
            similarity_score_cosine = compute_similarity(feature_img_exp, 
                                                         feature_img_tar, metric='cosine')

            # Store the similarity score and the coordinates
            coords.append((x, y))
            gabor_list.append(similarity_score_cosine)

    # Pair the similarity scores with coordinates
    cos_with_coords = list(zip(gabor_list, coords))

    # Sort based on cosine similarity (descending order for highest similarity)
    cos_with_coords.sort(key=lambda x: x[0], reverse=True)

    # Get the top 3 most similar patches
    largest_cos = cos_with_coords[:3]

    # Highlight the most similar regions in the output image
    for _, (x, y) in largest_cos:
        gabor_out[y:y + kernel_size, x:x + kernel_size] = 0

    # Find contours for the most similar regions
    inverted_mask = cv2.bitwise_not(gabor_out)
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the most similar regions
    img_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Display the most similar region
    ax[1].imshow(img_rgb)
    ax[1].axis("off")
    ax[1].set_title('Most similar region')

    plt.show()

    return largest_cos

#######################
def calculate_lbp_histogram(image, num_points=8, radius=1):
    """
    Calculate the Local Binary Pattern (LBP) of an image and return 
    the LBP image and normalized histogram.

    Parameters
    ----------
    image : np.ndarray
        The input image (can be grayscale or color).
    num_points : int, optional
        Number of circularly symmetric neighbor set points (default is 8).
    radius : int, optional
        Radius of the circle for LBP computation (default is 1).

    Returns
    -------
    lbp : np.ndarray
        The LBP-transformed image.
    lbp_hist : np.ndarray
        The normalized LBP histogram.
    """

    # Check if the input is a valid image
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid image (numpy array).")

    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:  # Color image (3 channels)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image  # Already grayscale

    # Calculate the LBP using the 'uniform' method
    lbp = local_binary_pattern(image_gray, num_points, radius, method='uniform')

    # Calculate the LBP histogram
    lbp_hist, _ = np.histogram(lbp.ravel(), 
                               bins=np.arange(0, num_points + 3), range=(0, num_points + 2))

    # Normalize the histogram
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize with a small epsilon to avoid division by zero

    return lbp, lbp_hist

def chi_square_distance(hist1, hist2):
    """
    Compute the chi-square distance between two histograms.

    Parameters
    ----------
    hist1 : np.ndarray
        The first histogram (must be non-negative).
    hist2 : np.ndarray
        The second histogram (must be non-negative).

    Returns
    -------
    float
        The chi-square distance between the two histograms.
    """

    # Validate input shapes
    if hist1.shape != hist2.shape:
        raise ValueError("Histograms must have the same shape.")

    # Ensure the histograms are non-negative
    if np.any(hist1 < 0) or np.any(hist2 < 0):
        raise ValueError("Histograms must contain non-negative values.")

    # Compute the chi-square distance with a small epsilon to prevent division by zero
    return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-7))

def lbp_similarity(image1, image2, num_points=8, radius=1):
    """
    Calculate the similarity between two images based on their LBP histograms.
    It returns the Chi-square and Euclidean distance between the LBP histograms of the two images.

    Parameters
    ----------
    image1 : np.ndarray
        The first input image.
    image2 : np.ndarray
        The second input image.
    num_points : int, optional
        The number of points to consider in LBP (default is 8).
    radius : int, optional
        The radius for LBP calculation (default is 1).

    Returns
    -------
    chi_square_dist : float
        The Chi-square distance between the LBP histograms of the two images.
    euclidean_dist : float
        The Euclidean distance between the LBP histograms of the two images.
    """

    # Ensure images are grayscale
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate LBP histograms for both images
    _, hist1 = calculate_lbp_histogram(image1, num_points=num_points, radius=radius)
    _, hist2 = calculate_lbp_histogram(image2, num_points=num_points, radius=radius)

    # Calculate the chi-square and euclidean distance between the two histograms
    chi_square_dist = chi_square_distance(hist1, hist2)
    euclidean_dist = euclidean(hist1, hist2)

    return chi_square_dist, euclidean_dist


def chi_lbp_pattern_match(df, idx, exp_img, kernel_size=300, overlap=None, hsv=None):
    """
    Compare regions in an image with a given experimental image using 
    LBP histograms and Chi-square distances.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image metadata (paths, timestamps, etc.).
    idx : int
        Index of the image in the DataFrame to be analyzed.
    exp_img : np.ndarray
        The experimental image to compare with.
    kernel_size : int, optional
        Size of the patches to be extracted (default is 300).
    overlap : int, optional
        Stride or overlap for sliding window (default is None for no overlap).
    hsv : bool, optional
        Whether to use the saturation channel of the image (default is None, uses grayscale).

    Returns
    -------
    tuple
        A tuple containing the coordinates of the smallest Chi-square patches and the 
        average Chi-square distance.
    """

    # Get timestamp and other metadata
    _time = df.loc[idx, "Timestamp"]
    title_text = f"{df.loc[idx, 'crop_type']} IMG {_time.year}/{_time.month}/{_time.day} " \
                 f"| {check_crop_stage(df.loc[idx, 'crop_type'], _time.month)} stage"

    # Load the image and convert to necessary formats
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)
    im_hsv = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HSV) if hsv else None

    # If hsv is enabled, use the saturation channel
    if hsv:
        _, s, _ = cv2.split(im_hsv)

    h, w = im_gray.shape

    # Resize experimental image to match kernel size
    exp_resized = cv2.resize(exp_img, (kernel_size, kernel_size))

    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(im_rgb)
    ax[0].axis("off")

    # Initialize output masks
    chi_out = np.full_like(im_gray, 255)

    chi_list = []
    coords = []

    # Set stride for sliding window (overlap or no overlap)
    stride = overlap if overlap else kernel_size

    # Sliding window over the image
    for y in range(0, h - kernel_size + 1, stride):
        for x in range(0, w - kernel_size + 1, stride):
            # Extract the patch using the current window position
            if hsv:
                patch = s[y:y + kernel_size, x:x + kernel_size]
            else:
                patch = im_gray[y:y + kernel_size, x:x + kernel_size]
            # Calculate LBP similarity (Chi-square distance)
            chi_v, _ = lbp_similarity(exp_resized, patch)
            coords.append((x, y))
            chi_list.append(chi_v)

    # Sort by smallest chi-square distance
    chi_with_coords = list(zip(chi_list, coords))
    chi_with_coords.sort(key=lambda x: x[0])

    # Get the smallest three values
    smallest_chi = chi_with_coords[:3]  # Get the three smallest patches

    # Set the corresponding regions in the chi-square output to zero (to highlight)
    for _, (x, y) in smallest_chi:
        chi_out[y:y + kernel_size, x:x + kernel_size] = 0

    # Find contours of the highlighted regions
    inverted_mask = cv2.bitwise_not(chi_out)
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes around the most similar regions
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box

        # Only draw on the saturation image if hsv is True
        if hsv:
            cv2.rectangle(s, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Plot results
    ax[1].imshow(img_rgb)
    ax[1].axis("off")
    ax[1].set_title('Most similar region')

    if hsv:
        ax[2].imshow(s, cmap="gray")
        ax[2].axis("off")
        ax[2].set_title("Saturation Channel")
    else:
        ax[2].axis("off")

    # Title and layout adjustments
    fig.suptitle(title_text, fontsize=18, y=0.8)
    fig.subplots_adjust(top=0.85, wspace=0.05)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    return smallest_chi, np.mean(chi_list)
########################

def hellinger_distance(hist1, hist2):
    """
    Compute the Hellinger distance between two histograms.

    Parameters
    ----------
    hist1 : np.ndarray
        The first histogram (must be a probability distribution).
    hist2 : np.ndarray
        The second histogram (must be a probability distribution).

    Returns
    -------
    float
        The Hellinger distance between the two histograms.
    """
    return np.sqrt(np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2)) / np.sqrt(2)

def hellinger_similarity(img1, img2):
    """
    Compute the Hellinger similarity between two images using LBP histograms.

    Parameters
    ----------
    img1 : np.ndarray
        First input image (grayscale).
    img2 : np.ndarray
        Second input image (grayscale).

    Returns
    -------
    float
        The Hellinger distance between the LBP histograms of the two images.
    """
    # Ensure both images are in grayscale
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute LBP histograms for both images
    _, hist1 = calculate_lbp_histogram(img1)
    _, hist2 = calculate_lbp_histogram(img2)

    # Return the Hellinger distance between the two histograms
    return hellinger_distance(hist1, hist2)

def helldist_lbp_pattern_match(df, idx, exp_img, kernel_size=300, overlap=None, hsv=None):
    """
    Find the most similar region in an image compared to a reference pattern
    using Hellinger distance on LBP histograms.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing metadata and image paths.
    idx : int
        The index of the DataFrame row to use for matching.
    exp_img : np.ndarray
        The reference image or pattern to compare with.
    kernel_size : int, optional
        The size of the kernel to extract patches from the target image. Default is 300.
    overlap : int, optional
        The stride or overlap value for patch extraction. If not provided, non-overlapping patches will be used.
    hsv : bool, optional
        If True, use the Saturation channel from the HSV color space for comparison. Otherwise, use grayscale.

    Returns
    -------
    tuple
        A tuple containing the most similar patch's coordinates and the average similarity score across all patches.
    """

    # Extract image metadata from DataFrame
    _time = df.loc[idx, "Timestamp"]
    title_text = f"{df.loc[idx, 'crop_type']} IMG {_time.year}/{_time.month}/{_time.day}"\
                                f"| {check_crop_stage(df.loc[idx, 'crop_type'], _time.month)} stage"
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")

    # Load the image and handle potential errors
    imgdata = cv2.imread(path)
    if imgdata is None:
        raise FileNotFoundError(f"Image not found at {path}")

    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)
    im_hsv = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HSV)

    # Choose the channel to work with
    if hsv:
        _, s, _ = cv2.split(im_hsv)
    else:
        s = im_gray

    # Resize the experimental image to match the kernel size
    exp_resized = cv2.resize(exp_img, (kernel_size, kernel_size))

    h, w = s.shape

    # Ensure kernel size is not larger than the image
    if kernel_size > h or kernel_size > w:
        raise ValueError("Kernel size is larger than the image dimensions.")

    # Initialize output mask and variables
    helldist_out = np.full_like(s, 255)
    dist_list = []
    coords = []

    # Extract patches with or without overlap
    if overlap:
        stride = overlap
        for y in range(0, h - kernel_size + 1, stride):
            for x in range(0, w - kernel_size + 1, stride):
                patch = s[y:y + kernel_size, x:x + kernel_size]
                sim_v = hellinger_similarity(exp_resized, patch)
                coords.append((x, y))
                dist_list.append(sim_v)
    else:
        for y in range(0, h, kernel_size):
            for x in range(0, w, kernel_size):
                if (x + kernel_size <= w) and (y + kernel_size <= h):
                    patch = s[y:y + kernel_size, x:x + kernel_size]
                    sim_v = hellinger_similarity(exp_resized, patch)
                    coords.append((x, y))
                    dist_list.append(sim_v)

    # Pair similarities with coordinates and sort by similarity (smallest distance)
    chi_with_coords = list(zip(dist_list, coords))
    chi_with_coords.sort(key=lambda x: x[0])

    # Get the most similar region
    smallest_sim = chi_with_coords[:1]  # Get the smallest similarity value and coordinates

    for _, (x, y) in smallest_sim:
        helldist_out[y:y + kernel_size, x:x + kernel_size] = 0  # Set the corresponding region to zero

    # Find contours for the most similar region
    inverted_mask = cv2.bitwise_not(helldist_out)
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the RGB image to BGR for OpenCV
    img_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes around the most similar regions
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
        if hsv:
            cv2.rectangle(s, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(im_rgb)
    ax[0].axis("off")
    ax[0].set_title('Original Image')

    ax[1].imshow(img_rgb)
    ax[1].axis("off")
    ax[1].set_title('Most Similar Region')

    if hsv:
        ax[2].imshow(s, cmap="gray")
        ax[2].axis("off")
        ax[2].set_title("Saturation Channel")
    else:
        ax[2].axis("off")

    fig.suptitle(title_text, fontsize=18, y=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return smallest_sim, np.mean(dist_list)
