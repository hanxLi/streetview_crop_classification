import cv2
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np

from utils import *

def lab_split(df, idx):
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    time = df.loc[idx, "Timestamp"]
    
    title_text = f"{df.loc[idx, 'crop_type']} IMG {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"

    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_lab = cv2.cvtColor(imgdata, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(im_lab)
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].set_title("RGB")
    ax[0, 0].axis('off')
    ax[0, 0].imshow(im_rgb)


    ax[0, 1].set_title(f"L Channel [{np.min(l)} - {np.max(l)}]")
    ax[0, 1].axis('off')
    ax[0, 1].imshow(l, cmap='gray')

    ax[1, 0].set_title(f"A Channel [{np.min(a)} - {np.max(a)}]")
    ax[1, 0].axis('off')
    ax[1, 0].imshow(a, cmap='gray')

    ax[1, 1].set_title(f"B Channel [{np.min(b)} - {np.max(b)}]")
    ax[1, 1].axis('off')
    ax[1, 1].imshow(b, cmap='gray')

    fig.suptitle(title_text, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 

    plt.show()

    return l, a, b

def hsv_split(df, idx):

    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    time = df.loc[idx, "Timestamp"]
    
    title_text = f"{df.loc[idx, 'crop_type']} IMG {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"

    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_hsv = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(im_hsv)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].set_title("RGB")
    ax[0, 0].axis('off')
    ax[0, 0].imshow(im_rgb)


    ax[0, 1].set_title(f"H Channel [{np.min(h)} - {np.max(h)}]")
    ax[0, 1].axis('off')
    ax[0, 1].imshow(h, cmap='gray')

    ax[1, 0].set_title(f"S Channel [{np.min(s)} - {np.max(s)}]")
    ax[1, 0].axis('off')
    ax[1, 0].imshow(s, cmap='gray')

    ax[1, 1].set_title(f"V Channel [{np.min(v)} - {np.max(v)}]")
    ax[1, 1].axis('off')
    ax[1, 1].imshow(v, cmap='gray')

    fig.suptitle(title_text, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 

    # plt.show()

    return h, s, v

def hls_split(df, idx):

    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    time = df.loc[idx, "Timestamp"]
    
    title_text = f"{df.loc[idx, 'crop_type']} IMG {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"

    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_lab = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HLS)


    plt.title(title_text)
    plt.axis('off')
    plt.imshow(im_rgb)
    plt.show()

    plt.title("HLS")
    plt.axis('off')
    plt.imshow(im_lab)
    plt.show()

    h, l, s = cv2.split(im_lab)
    plt.title(f"H Channel [{np.min(h)} - {np.max(h)}]")
    plt.axis('off')
    plt.imshow(h, cmap='gray')
    plt.show()

    plt.title(f"L Channel [{np.min(l)} - {np.max(l)}]")
    plt.axis('off')
    plt.imshow(l, cmap='gray')
    plt.show()

    plt.title(f"S Channel [{np.min(s)} - {np.max(s)}]")
    plt.axis('off')
    plt.imshow(s, cmap='gray')
    plt.show()

    return h, l, s

def mask_with_hsv(df, idx, h_thresh_range = [40, 90], s_thresh_range = [120, 255]):
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    time = df.loc[idx, "Timestamp"]
    
    title_text = f"{df.loc[idx, 'crop_type']} | {df.loc[idx, 'Name']} | {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"

    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_hsv = cv2.cvtColor(imgdata, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(im_hsv)

    range_h = cv2.inRange(h, h_thresh_range[0], h_thresh_range[1])
    range_s = cv2.inRange(s, s_thresh_range[0], s_thresh_range[1])
    mask = np.logical_and(range_h, range_s).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Display the first image in the first subplot
    ax[0].imshow(im_rgb)
    ax[0].set_title('RGB Original IMG')
    ax[0].axis('off')  # Turn off axis labels

    # Display the second image in the second subplot
    ax[1].imshow(mask)
    ax[1].set_title('Crop Mask from HSV Transform')
    ax[1].axis('off')  # Turn off axis labels

    # Show the plot
    fig.suptitle(title_text)
    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.show()

def mask_with_lab(df, idx, a_thresh = 120, b_thresh = 140, soybean=False):
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    time = df.loc[idx, "Timestamp"]
    
    title_text = f"{df.loc[idx, 'crop_type']} | {df.loc[idx, 'Name']} | {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"

    imgdata = cv2.imread(path)
    im_rgb = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
    im_lab = cv2.cvtColor(imgdata, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(im_lab)

    thresh_a = cv2.threshold(a, a_thresh, 255, cv2.THRESH_BINARY_INV)[1]
    thresh_b = cv2.threshold(b, b_thresh, 255, cv2.THRESH_BINARY_INV)[1]

    if soybean:
        thresh_b = np.logical_not(thresh_b).astype(np.uint8)
        mask = np.logical_and(thresh_a, thresh_b).astype(np.uint8)
    else:
        mask = (~(thresh_a == 0) | ~(thresh_b == 0)).astype(np.uint8)
        mask = np.logical_not(mask).astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Display the first image in the first subplot
    ax[0].imshow(im_rgb)
    ax[0].set_title('RGB Original IMG')
    ax[0].axis('off')  # Turn off axis labels

    # Display the second image in the second subplot
    ax[1].imshow(mask)
    ax[1].set_title('Crop Mask from LAB Transform')
    ax[1].axis('off')  # Turn off axis labels

    # Show the plot
    fig.suptitle(title_text)
    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.show()

def albedo_convert(df, idx, threshold):
    path = df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
    time = df.loc[idx, "Timestamp"]
    title_text = f"{df.loc[idx, 'crop_type']} | {df.loc[idx, 'Name']} | {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage"

    image = cv2.imread(path)
    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the R, G, and B channels to the range [0, 1]
    image_normalized = image_rgb / 255.0

    # Calculate the reflectance using the given formula: (R/255 + G/255 + B/255) / 3
    reflectance = np.mean(image_normalized, axis=2)  # Average across the R, G, B channels

    fig, ax = plt.subplots(1, 2, figsize = (15, 10))

    # Display the RGB Image
    ax[0].imshow(image_rgb)
    ax[0].set_title("RGB Image")
    ax[0].axis("off")

    # Display the reflectance Image
    ax[1].imshow(reflectance < threshold, cmap='gray')
    ax[1].set_title("Reflectance Thresholded Img")
    ax[1].axis("off")

    fig.suptitle(title_text, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
