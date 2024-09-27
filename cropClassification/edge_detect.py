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

def canny_edge(df, df_index, threshold=None, enhance=False):
    time = df.loc[df_index, "Timestamp"]
    title_text = f"{df.loc[df_index, 'crop_type']} | {df.loc[df_index, 'Name']} | {time.year}/{time.month}/{time.day} | {check_crop_stage(df.loc[df_index, 'crop_type'], time.month)} stage"

    imgdata = cv2.imread(df.loc[df_index, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/"))
    im_gray = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)
    
    img_copy = im_gray.copy()
    if enhance:
        img_copy = img_copy * 3
    img_blur = cv2.GaussianBlur(img_copy, (5, 5), 0)
    
    if threshold:
        upper_threshold = threshold[1]
        lower_threshold = threshold[0]
    else:
        otsu_thresh, _ = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_threshold = otsu_thresh * 0.5
        upper_threshold = otsu_thresh * 1.5
    
    edges = cv2.Canny(img_blur, lower_threshold, upper_threshold)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].axis("off")
    ax[0].set_title("Gray Scale")
    ax[0].imshow(im_gray, cmap="gray")


    ax[1].axis("off")
    ax[1].set_title("Edge")
    ax[1].imshow(edges, cmap="gray")

    fig.suptitle(title_text, fontsize=16)
    plt.tight_layout() 
    plt.show()
    return edges

def calc_edge_pair(img, threshold=None, enhance=False):
    
    img_copy = img.copy()
    if enhance:
        img_copy = img_copy * 3
    img_blur = cv2.GaussianBlur(img_copy, (5, 5), 0)
    
    if threshold:
        upper_threshold = threshold[1]
        lower_threshold = threshold[0]
    else:
        otsu_thresh, _ = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_threshold = otsu_thresh * 0.5
        upper_threshold = otsu_thresh * 1.5
    
    edges = cv2.Canny(img_blur, lower_threshold, upper_threshold)

    plt.axis("off")
    plt.imshow(img_copy, cmap="gray")
    plt.show()

    plt.axis("off")
    plt.title(f"[{np.min(edges)} - {np.max(edges)}]")
    plt.imshow(edges, cmap="gray")
    plt.show()
    return edges