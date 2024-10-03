# Standard library imports
import os
import random
from pathlib import Path
from PIL import Image
from IPython.display import display, clear_output

# Third-party libraries
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm.notebook import tqdm

# Local imports (project-specific modules)
from utils import check_crop_stage
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"



#######################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for task purposes
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

def show_best_masks(image, masks, scores, show_plot = True, point_coords=None, box_coords=None, input_labels=None, borders=True, title=None):
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
    
    if show_plot:
        # Display the image with the mask that has the highest score

        fig, ax = plt.subplots(1, 2, figsize=(15, 10))


        ax[0].imshow(image)
        ax[0].set_title("Original Image")
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

    return max_mask, max_score

#######################################

def get_three_click_coordinates(image_path, current_ct, total_ct):
    """
    Captures three user-defined points by clicking on an image.

    This function opens an image from the specified path in an OpenCV window, allowing the user to click on three points.
    The coordinates of the clicked points are recorded and returned.

    Parameters:
    -----------
    image_path : str
        The file path to the image on which the user will click to select points.

    Returns:
    --------
    list of tuple
        A list of three (x, y) coordinate tuples representing the clicked points. 
        Returns None if the image cannot be loaded.

    Notes:
    ------
    - The user can click on three points in the image, and the function will 
        close automatically once three points are selected.
    - Pressing the ESC key will exit the window prematurely.
    """

    img = cv2.imread(image_path)

    # Check if the image is loaded properly
    if img is None:
        print(f"Error: Unable to open image at {image_path}")
        return None

    coordinates = []

    def mouse_callback(event, x, y, _flags, _param):
        nonlocal coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            # print(f"Click at: ({x}, {y})")
            if len(coordinates) == 3:
                cv2.setMouseCallback(winname, lambda *args: None)  # Disable further clicks

    # Create a named window and show the image
    winname = f"({current_ct}/{total_ct}) | Left Click on 3 Points | ESC to exit"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL to allow resizing

    cv2.startWindowThread()

    cv2.imshow(winname, img)

    # Set the mouse callback to capture clicks
    cv2.setMouseCallback(winname, mouse_callback)

    # Use a loop to keep the window open and responsive
    while True:
        # Display the image and wait for a key or three clicks
        key = cv2.waitKey(1) & 0xFF  # Wait for 1 ms and check for ESC key

        if key == 27 or len(coordinates) == 3:  # Exit on 'ESC' key or after 3 clicks
            break
    # Close all windows and end the program
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return coordinates 

def get_mask_prompt(img, pixels_below=None, num_points=3):
    """
    Generates mask prompt points below the skyline of an image.

    The function identifies the skyline in the input image by detecting the 
      bottom-most sky pixel in each column. 
    It then selects a specified number of points below the skyline and returns 
      their coordinates.

    Parameters:
    -----------
    img : np.array
        The input image in RGB format.
    pixels_below : list, optional
        A range [min, max] to define how far below the skyline the points should be selected. 
        Default is [100, 300].
    num_points : int, optional
        The number of points to return. Default is 3.

    Returns:
    --------
    list
        A list of coordinates [[x1, y1], [x2, y2], ...] representing the points below the skyline. If no valid points 
        are found, the center of the image is returned.
    """
    if pixels_below is None:
        pixels_below = [100, 300]
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

def sam2_mask_generate(df_path, idx, model_params, point = None):
    """
    Generates segmentation masks for an image using a SAM2 model and visualizes the best masks.

    This function reads geospatial data from a GeoJSON file, retrieves an image
      associated with the given index, processes it using a SAM2 model, 
      generates segmentation masks based on input points or a default prompt, 
      and then displays the best masks based on prediction scores.

    Parameters:
    -----------
    df_path : str
        The file path to the GeoJSON file containing geospatial data.
    idx : int
        The index of the row in the DataFrame to extract image data from.
    model_params : tuple
        A tuple containing three elements:
            - chkpt (str): Path to the model checkpoint file.
            - cfg_path (str): Path to the configuration file for the SAM2 model.
            - device (str): The device type on which to run the model, e.g., 'cuda' or 'cpu'.
    point : list, optional
        A list of coordinates for generating the mask. If None, a default point prompt is generated. 
        The list should contain points in the format [[x1, y1], [x2, y2], ...]. (default is None)

    Returns:
    --------
    None
        The function doesn't return any values but displays the best segmentation masks based on prediction scores.
    
    Exceptions:
    -----------
    - The function may raise exceptions if file paths are incorrect, or if the model fails to load due to incorrect parameters.
    """
    df = gpd.read_file(df_path, driver="GEOJSON")

    image = Image.open(df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/"))
    image = np.array(image.convert("RGB"))
    time = df.loc[idx, "Timestamp"]

    title_text = (
        f"{df.loc[idx, 'crop_type']} | {df.loc[idx, 'Name']} | "
        f"{time.year}/{time.month}/{time.day} | "
        f"{check_crop_stage(df.loc[idx, 'crop_type'], time.month)} stage")

    h, v, _ = image.shape

    chkpt, cfg_path, device = model_params
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    sam2_model = build_sam2(cfg_path, chkpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    if not point:
        point = get_mask_prompt(image)

    lbl_list = [1 for _ in range(len(point))]

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

    show_best_masks(image, masks, scores, show_plot=True, point_coords=input_point,
                    input_labels=input_label, borders=True, title=title_text)   

def sam2_manual_masking(df, model_params, save_path, show_plot=False, return_df=False):
    """
    Performs manual masking on images using a SAM2 model and saves the best mask for each image.

    This function processes a DataFrame of image metadata, applies 
    the SAM2 model to generate segmentation masks for each image using 
    manually selected points, and saves the best mask image to the specified save path.

    It also records the mask score for each image in the returned DataFrame.
    Only rows with a valid mask score will be included in the returned DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing metadata of images. It must include the following columns:
        - 'save_path': The file path of the image.
        - 'Name': The name of the image file.
        - 'Timestamp': A timestamp indicating when the image was taken.
        - 'crop_type': The type of crop present in the image.
    model_params : tuple
        A tuple containing three elements:
            - chkpt (str): Path to the SAM2 model checkpoint file.
            - cfg_path (str): Path to the configuration file for the SAM2 model.
            - device (str): The device type on which to run the model, e.g., 'cuda' or 'cpu'.
    save_path : str or Path
        The directory path where the generated mask images should be saved.
    show_plot : boolean
        If set to True, will plot the image and mask pair in console. Default is False.
    return_df : boolean, optional
        If True, the function will return the filtered DataFrame with valid mask scores.


    Returns:
    --------
    pandas.DataFrame
        A copy of the original DataFrame with an additional column 'mask_score'
        indicating the score of the best mask for each image.
        Only rows with valid mask scores (non-NaN) are returned.
    """
    df_copy = df.copy()
    df_copy['mask_score'] = np.nan  # Initialize mask score column

    for i in range(len(df)):
        path = df.loc[i, "save_path"].replace("/home/hanxli/data/",
                                              "/Users/steeeve/Documents/csiss/")
        img_name = f"{df.loc[i, 'Name'].split('.')[0]}_mask.jpg"
        mask_path = Path(save_path) / img_name

        if mask_path.exists():
            # print(f"Mask for '{df.loc[i, 'Name']}' already exists. Skipping...")
            continue

        image = Image.open(path)
        image = np.array(image.convert("RGB"))
        time = df.loc[i, "Timestamp"]

        title_text = (
            f"{df.loc[i, 'crop_type']} | {df.loc[i, 'Name']} | "
            f"{time.year}/{time.month}/{time.day} | "
            f"{check_crop_stage(df.loc[i, 'crop_type'], time.month)} stage")
        h, v, _ = image.shape

        chkpt, cfg_path, device = model_params
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        sam2_model = build_sam2(cfg_path, chkpt, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)

        # Get manual points, skip if no points selected
        point = get_three_click_coordinates(path, i, len(df))
        if not point:  # If point list is empty, skip this iteration
            continue

        # Generate labels and prepare input points for predictor
        lbl_list = [1 for _ in range(len(point))]
        point.append([v // 2, h - 300])
        lbl_list.append(0)

        input_point = np.array(point)
        input_label = np.array(lbl_list)

        # Predict masks based on points
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True)

        # Sort masks by score
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # Display and save the best mask
        mask, score = show_best_masks(image, masks, scores,
                                      show_plot=show_plot,
                                      point_coords=input_point,
                                      input_labels=input_label,
                                      borders=True,
                                      title=title_text)
        cv2.imwrite(str(mask_path), mask)

        # Save the mask score in the DataFrame
        df_copy.at[i, "mask_score"] = score
        df_copy.at[i, "mask_name"] = img_name
        df_copy.at[i, "mask_path"] = str(mask_path).replace("/Users/steeeve/Documents/csiss/",
                                                       "/home/hanxli/data/")

    # Filter the DataFrame to return only rows with valid mask scores (non-NaN)
    df_valid = df_copy.dropna(subset=["mask_score"])
    geojson_path = os.path.join(save_path, "valid_img_mask.geojson")
    if Path(geojson_path).exists():
        # Load existing GeoDataFrame
        existing_gdf = gpd.read_file(geojson_path)
        print(f"Existing GeoDataFrame loaded with {len(existing_gdf)} entries.")
        
        # Append new mask data to the existing GeoDataFrame
        df_valid = pd.concat([existing_gdf, df_valid], ignore_index=True)
        df_valid = gpd.GeoDataFrame(df_valid)
    else:
        # Convert df_valid to GeoDataFrame if not appending to an existing file
        df_valid = gpd.GeoDataFrame(df_valid)

    # Save the updated GeoDataFrame
    df_valid.to_file(geojson_path, driver="GeoJSON")
    print(f"GeoDataFrame saved with {len(df_valid)} total entries.")

    if return_df:
        return df_valid


def load_and_display_img_mask_pair(img_path, mask_path, mask_score, current, total):
    """
    Load an image and its corresponding mask, apply the mask as an overlay, and display them.

    Parameters
    ----------
    img_path : str
        The file path of the original image.
    mask_path : str
        The file path of the mask image.
    mask_score : float or str
        The score or identifier for the mask, used in the plot title.
    current : int
        The current image-mask pair number in the evaluation process.
    total : int
        The total number of image-mask pairs to be evaluated.

    Returns
    -------
    None
        This function only displays plots and does not return any values.
    """
    try:
        original_image = Image.open(img_path)
        mask_image = Image.open(mask_path).convert('L')  # Convert mask to grayscale
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading image or mask: {e}")
        return

    # Convert images to numpy arrays for manipulation
    original_np = np.array(original_image)
    mask_np = np.array(mask_image)

    # Ensure mask is binary (0 and 1)
    mask_np = (mask_np > 0).astype(np.uint8)

    # Create a copy of the original image to apply the mask
    image_with_mask = original_np.copy()

    # Apply red color where the mask is 1
    image_with_mask[mask_np == 1, 0] = 255  # Red channel
    image_with_mask[mask_np == 1, 1] = 0    # Green channel
    image_with_mask[mask_np == 1, 2] = 0    # Blue channel

    # Plot the images side by side
    plt.figure(figsize=(20, 10))

    # Set the common title for the entire figure
    plt.suptitle(f"Evaluate SAM Generated Mask({current + 1}/{total})", fontsize=16)
    

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_np)
    plt.title("Original Image")
    plt.axis("off")

    # Plot the image with mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(original_np)
    plt.imshow(image_with_mask, alpha=0.4)  # Apply 60% transparency to the mask overlay
    plt.title(f"Image with Mask Overlay | Mask Score: {mask_score}")
    plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Display the figure
    display(plt.gcf())
    plt.close()



def evaluate_img_mask_pair(df, mask_save_path, return_df=False, continue_index=None):
    """
    Evaluate image and mask pairs, allowing user interaction to accept or reject each pair.

    The function now checks the mask save directory to identify previously deleted masks,
    and resumes evaluation accordingly. It accepts 'y' or 'n' input from the user to accept
    or reject each image-mask pair.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing image and mask information. The following columns are expected:
        - 'save_path': The path to the image.
        - 'mask_name': The name of the mask file.
        - 'mask_score': The score of the mask (can be NaN).
    mask_save_path : str
        The directory path where mask files are stored, and where the output GeoJSON files will be saved.
    return_df : bool, optional, default=False
        If True, the function returns the modified DataFrame with accepted image-mask pairs.
    continue_index : int, optional
        The index to resume evaluation from. If provided, the function will reconstruct the
        accepted and deleted mask lists up to this point based on the current state of the mask directory.

    Returns
    -------
    pandas.DataFrame or None
        Returns the modified DataFrame with accepted image-mask pairs if return_df is True. Otherwise, returns None.
    """

    # List to keep track of accepted and deleted masks
    accepted_indices = []
    deleted_masks = []

    # Step 1: Check the mask save directory to see which masks are already deleted
    existing_masks = set(os.listdir(mask_save_path))  # List all mask files currently in the directory

    # Step 2: Rebuild the accepted and deleted lists by checking the current state of the masks
    for idx, row in df.iterrows():
        mask_name = row["mask_name"]
        mask_path = os.path.join(mask_save_path, mask_name)
        
        # If the mask file exists in the directory, it hasn't been deleted yet
        if mask_name in existing_masks:
            accepted_indices.append(idx)  # Consider it accepted so far
        else:
            deleted_masks.append(mask_path)  # Track the mask as already deleted

    # Step 3: Resume from the continue_index if provided
    start_index = 0
    if continue_index is not None and continue_index < len(df):
        start_index = continue_index
        # Adjust the accepted indices to only include those before the continue index
        accepted_indices = [idx for idx in accepted_indices if idx < start_index]
    
    # Step 4: Continue the evaluation of images from the continue_index or start
    for i in range(start_index, len(df)):
        row = df.iloc[i]
        img_path = row["save_path"].replace("/home/hanxli/data/",
                                            "/Users/steeeve/Documents/csiss/")
        mask_name = row["mask_name"]
        mask_path = os.path.join(mask_save_path, mask_name)
        mask_score = row["mask_score"]

        # Check if the mask path exists
        if mask_name not in existing_masks:
            print(f"Mask file '{mask_name}' was already deleted. Skipping...")
            continue

        # Clear previous plot output
        clear_output(wait=True)

        # Use the helper function to load and display the image-mask pair with progress
        load_and_display_img_mask_pair(img_path, mask_path, mask_score, i, len(df))

        # Prompt user for input (y/n)
        user_input = input("Accept with 'y' or reject and delete with 'n' (y/n): ").strip().lower()

        if user_input == 'y':
            accepted_indices.append(i)  # Use the positional index `i` directly
        elif user_input == 'n':
            try:
                os.remove(mask_path)
                print(f"Mask file '{mask_name}' deleted.")
                deleted_masks.append(mask_path)
                existing_masks.remove(mask_name)  # Update existing masks to reflect the deletion
            except OSError as e:
                print(f"Error deleting mask file: {e}")

    # Step 5: After review, update the DataFrame to exclude deleted masks using iloc (positional indices)
    df = df.iloc[accepted_indices].reset_index(drop=True)  # Use the original DataFrame index
    df_new = df.copy()
    df = gpd.GeoDataFrame(df, geometry="geometry")
    orig_df_save_path = os.path.join(mask_save_path, "valid_img_mask.geojson")
    df.to_file(orig_df_save_path, driver="GeoJSON")

    # Rename columns as needed
    df_new = df_new[["Name", "crop_type", "mask_name", "save_path", "mask_path",
                     "mask_score", "Timestamp", "geometry"]].rename(columns={"Name": "img_name",
                                                                             "save_path": "img_path",
                                                                             "Timestamp": "time"})

    # Ensure df is a GeoDataFrame before saving
    df_new = gpd.GeoDataFrame(df_new, geometry="geometry")

    # Save the GeoDataFrame to a GeoJSON file
    geojson_path = os.path.join(mask_save_path, "evaluated_img_mask_pair.geojson")
    df_new.to_file(geojson_path, driver="GeoJSON")

    print(f"\nEvaluation complete. {len(deleted_masks)} mask files were deleted.")
    if return_df:
        return df




def show_img_mask_pair(df, path):
    """
    Displays image and mask pairs side by side with mask overlay applied on the original image.
    """
    for i, row in df.iterrows():
        img_path = row["save_path"].replace("/home/hanxli/data/",
                                            "/Users/steeeve/Documents/csiss/")
        mask_path = os.path.join(path, row["mask_name"])
        mask_score = row["mask_score"]

        # Check if the mask path exists
        if not os.path.exists(mask_path):
            print(f"Mask file '{mask_path}' does not exist. Skipping...")
            continue

        # Use the helper function to load and display the image-mask pair
        load_and_display_img_mask_pair(img_path, mask_path, mask_score, i, len(df))