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
    kernels = []
    # Create Gabor kernels with different orientations and frequencies
    ksize = 31  # Size of the filter
    for theta in np.arange(0, np.pi, np.pi / 8):  # Range of orientations
        for sigma in (3, 5, 7):  # Variance (sigma) of the gaussian envelope
            for lambd in np.arange(10, 30, 5):  # Wavelength of sinusoidal factor
                for gamma in (0.5, 1.0):  # Aspect ratio
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
    return kernels

# Step 3: Apply Gabor filters to the image
def apply_gabor_filters(image, kernels):
    filtered_images = []
    for kernel in kernels:
        filtered_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        filtered_images.append(filtered_img)
    return filtered_images

def extract_gabor_features(filtered_images):
    features = []
    for filtered_img in filtered_images:
        # Flatten the filtered image into a 1D feature vector
        features.append(filtered_img.flatten())
    # Combine all features into a single vector (or keep separate if needed)
    combined_feature = np.concatenate(features)
    return combined_feature

# Step 2: Compute similarity between two images (Gabor features)
def compute_similarity(features_img1, features_img2, metric='euclidean'):
    if metric == 'euclidean':
        # Compute Euclidean distance (smaller means more similar)
        return distance.euclidean(features_img1, features_img2)
    elif metric == 'cosine':
        # Compute Cosine similarity (1 means identical, 0 means completely different)
        return cosine_similarity([features_img1], [features_img2])[0][0]
    
def check_pattern_match_gabor(df, idx, exp_img, kernel_size = 300, overlap=None):
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)

    h, w = im_gray.shape

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(im_rgb)
    ax[0].axis("off")

    gabor_out = np.full_like(im_gray, 255)

    gabor_list = []
    coords = []

    
    kernels = build_gabor_kernels()
    exp_resized = cv2.resize(exp_img, (kernel_size, kernel_size))
    exp_filtered = apply_gabor_filters(exp_resized, kernels)
    feature_img_exp = extract_gabor_features(exp_filtered)
    
    
    # No overlaps

    # Overlaps
    if overlap:
        stride = overlap
        for y in range(0, h - kernel_size + 1, stride):
            for x in range(0, w - kernel_size + 1, stride):
                # Extract the patch using the current window position
                patch = im_gray[y:y + kernel_size, x:x + kernel_size]
                target_filtered = apply_gabor_filters(patch, kernels)
                feature_img_tar = extract_gabor_features(target_filtered)

                similarity_score_cosine = compute_similarity(feature_img_exp, feature_img_tar, metric='cosine')

                coords.append((x, y))
                gabor_list.append(similarity_score_cosine)
    else:
        for y in range(0, h, kernel_size):
            for x in range(0, w, kernel_size):
                # Extract the patch using the current window position
                if (x + kernel_size <= w) and (y + kernel_size <= h):
                    patch = im_gray[y:y + kernel_size, x:x + kernel_size]
                    target_filtered = apply_gabor_filters(patch, kernels)
                    feature_img_tar = extract_gabor_features(target_filtered)

                    similarity_score_cosine = compute_similarity(feature_img_exp, feature_img_tar, metric='cosine')

                    coords.append((x, y))
                    gabor_list.append(similarity_score_cosine)


    cos_with_coords = list(zip(gabor_list, coords))


    cos_with_coords.sort(key=lambda x: x[0])  


    # Get the smallest three values
    largest_cos = cos_with_coords[-3:]  # Get three smallest chi_v and their coordinates
    # smallest_euc = euc_with_coords[:3]  # Get three smallest euc_v and their coordinates

    for _, (x, y) in largest_cos:
        gabor_out[y:y + kernel_size, x:x + kernel_size] = 0  # Set the corresponding region to zero

    # for _, (x, y) in smallest_euc:
    #     euc_out[y:y + kernel_size, x:x + kernel_size] = 0  # Set the corresponding region to zero

    # Plotting chi_out and euc_out results

    inverted_mask = cv2.bitwise_not(gabor_out) 


    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the RGB image to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes around the contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box

    # Convert back to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ax[1].imshow(img_rgb)
    ax[1].axis("off")
    ax[1].set_title('Most similar region')

    plt.show()
    return largest_cos
    
#######################

# Function to calculate LBP and return its histogram
def calculate_lbp_histogram(image, num_points=8, radius=1):
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    
    # Calculate LBP
    lbp = local_binary_pattern(image_gray, num_points, radius, method='uniform')
    
    # Calculate the histogram of LBP
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    
    # Normalize the histogram
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    
    return lbp, lbp_hist

# Function to calculate chi-square distance manually
def chi_square_distance(hist1, hist2):
    return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-7))

# Function to calculate similarity between two images using LBP
def lbp_similarity(image1, image2):
    """
    This function calculates the similarity between two images based on LBP histograms.
    It returns the Chi-square and Euclidean distance between the LBP histograms of the two images.

    Parameters:
    - image1: First image
    - image2: Second image
    - num_points: Number of points to consider in LBP (default=8)
    - radius: Radius for LBP (default=1)

    Returns:
    - chi_square_dist: Chi-square distance between LBP histograms
    - euclidean_dist: Euclidean distance between LBP histograms
    """
    
    # Calculate LBP histograms for both images
    lbp1, hist1 = calculate_lbp_histogram(image1)
    lbp2, hist2 = calculate_lbp_histogram(image2)

    # plt.imshow(lbp1)
    # plt.show()

    # plt.imshow(lbp2)
    # plt.show()
    
    # Calculate the chi-square and euclidean distance between the two histograms
    chi_square_dist = chi_square_distance(hist1, hist2)
    euclidean_dist = euclidean(hist1, hist2)
    
    return chi_square_dist, euclidean_dist
            
def chi_lbp_pattern_match(df, idx, exp_img, kernel_size = 300, overlap=None, hsv=None):

    time = df.loc[idx, "Timestamp"]
    title_text = f"{df.loc[idx, 'crop_type']} IMG {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)
    im_hsv = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HSV)

    if hsv:
        _, s, _ = cv2.split(im_hsv)


    exp_resized = cv2.resize(exp_img, (kernel_size, kernel_size))


    h, w = im_gray.shape
    # h_exp, w_exp = exp_img.shape

    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(im_rgb)
    ax[0].axis("off")

    chi_out = np.full_like(im_gray, 255)

    chi_list = []
    coords = []

    # Overlaps
    if overlap:
        stride = overlap
        for y in range(0, h - kernel_size + 1, stride):
            for x in range(0, w - kernel_size + 1, stride):
                # Extract the patch using the current window position
                if hsv:
                    patch = s[y:y + kernel_size, x:x + kernel_size]
                    # patch = cv2.resize(patch, (h_exp, w_exp))
                else:
                    patch = im_gray[y:y + kernel_size, x:x + kernel_size]

                
                # chi_v, _ = lbp_similarity(exp_resized, patch)
                chi_v, _ = lbp_similarity(exp_img, patch)
                coords.append((x, y))
                chi_list.append(chi_v)
    else: # No overlaps
        for y in range(0, h, kernel_size):
            for x in range(0, w, kernel_size):
                # Extract the patch using the current window position
                if (x + kernel_size <= w) and (y + kernel_size <= h):
                    if hsv:
                        patch = s[y:y + kernel_size, x:x + kernel_size]
                    else:
                        patch = im_gray[y:y + kernel_size, x:x + kernel_size]
                    
                    # chi_v, _ = lbp_similarity(exp_resized, patch)
                    chi_v, _ = lbp_similarity(exp_img, patch)

                    coords.append((x, y))

                    chi_list.append(chi_v)

    chi_with_coords = list(zip(chi_list, coords))
    chi_with_coords.sort(key=lambda x: x[0])

    # Get the smallest three values
    smallest_chi = chi_with_coords[:2]  # Get three smallest chi_v and their coordinates

    for _, (x, y) in smallest_chi:
        chi_out[y:y + kernel_size, x:x + kernel_size] = 0  # Set the corresponding region to zero


    inverted_mask = cv2.bitwise_not(chi_out) 
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the RGB image to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes around the contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
        cv2.rectangle(s, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert back to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ax[1].imshow(img_rgb)
    ax[1].axis("off")
    ax[1].set_title('Most similar region')

    ax[2].imshow(s, cmap="gray")
    ax[2].axis("off")
    ax[2].set_title("Saturation Channel")


    fig.suptitle(title_text, fontsize=18, y=0.8)
    fig.subplots_adjust(top=0.85, wspace=0.05)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    # return smallest_chi, smallest_euc
    return smallest_chi, np.mean(chi_list)
########################

def hellinger_distance(hist1, hist2):
    """Compute the Hellinger distance between two histograms."""
    return np.sqrt(np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2)) / np.sqrt(2)

def hellinger_similarity(img1, img2):
    _, hist1 = calculate_lbp_histogram(img1)
    _, hist2 = calculate_lbp_histogram(img2)

    return hellinger_distance(hist1, hist2)

def helldist_lbp_pattern_match(df, idx, exp_img, kernel_size = 300, overlap=None, hsv=None):

    time = df.loc[idx, "Timestamp"]
    title_text = f"{df.loc[idx, 'crop_type']} IMG {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)
    im_hsv = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HSV)

    if hsv:
        _, s, _ = cv2.split(im_hsv)


    # exp_resized = cv2.resize(exp_img, (kernel_size, kernel_size))


    h, w = im_gray.shape
    # h_exp, w_exp = exp_img.shape

    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(im_rgb)
    ax[0].axis("off")

    helldist_out = np.full_like(im_gray, 255)

    dist_list = []
    coords = []

    # Overlaps
    if overlap:
        stride = overlap
        for y in range(0, h - kernel_size + 1, stride):
            for x in range(0, w - kernel_size + 1, stride):
                # Extract the patch using the current window position
                if hsv:
                    patch = s[y:y + kernel_size, x:x + kernel_size]
                    # patch = cv2.resize(patch, (h_exp, w_exp))
                else:
                    patch = im_gray[y:y + kernel_size, x:x + kernel_size]

                
                # chi_v, _ = lbp_similarity(exp_resized, patch)
                sim_v = hellinger_similarity(exp_img, patch)
                coords.append((x, y))
                dist_list.append(sim_v)
    else: # No overlaps
        for y in range(0, h, kernel_size):
            for x in range(0, w, kernel_size):
                # Extract the patch using the current window position
                if (x + kernel_size <= w) and (y + kernel_size <= h):
                    if hsv:
                        patch = s[y:y + kernel_size, x:x + kernel_size]
                    else:
                        patch = im_gray[y:y + kernel_size, x:x + kernel_size]
                    
                    # chi_v, _ = lbp_similarity(exp_resized, patch)
                    sim_v = hellinger_similarity(exp_img, patch)
                    coords.append((x, y))
                    dist_list.append(sim_v)


    chi_with_coords = list(zip(dist_list, coords))
    chi_with_coords.sort(key=lambda x: x[0])

    # Get the smallest three values
    smallest_sim = chi_with_coords[:1]  # Get smallest similarity value and their coordinates

    for _, (x, y) in smallest_sim:
        helldist_out[y:y + kernel_size, x:x + kernel_size] = 0  # Set the corresponding region to zero


    inverted_mask = cv2.bitwise_not(helldist_out) 
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the RGB image to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes around the contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
        if hsv:
            cv2.rectangle(s, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert back to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ax[1].imshow(img_rgb)
    ax[1].axis("off")
    ax[1].set_title('Most similar region')
    if hsv:
        ax[2].imshow(s, cmap="gray")
        ax[2].axis("off")
        ax[2].set_title("Saturation Channel")
    else:
        ax[2].axis("off")

    fig.suptitle(title_text, fontsize=18, y=0.8)
    fig.subplots_adjust(top=0.85, wspace=0.05)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    # return smallest_chi, smallest_euc
    return smallest_sim, np.mean(dist_list)


