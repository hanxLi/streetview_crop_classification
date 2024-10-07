import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.notebook import tqdm


# Function to convert an image to LAB, HSV, and stack them with RGB channels
def generate_stacked_image(image):
    """
    Convert an RGB image to a 9-channel image by stacking RGB, LAB, and HSV channels.
    LAB channels are clamped to avoid negative values in variance calculations.
    
    Args:
        image (np.ndarray): Input RGB image (H, W, 3).
    
    Returns:
        np.ndarray: 9-channel image (H, W, 9).
    """
    # Split the image into RGB channels
    b_channel, g_channel, r_channel = cv2.split(image)  # Note OpenCV loads as BGR by default
    
    # Convert to LAB color space and split into L, A, B channels
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_lab_channel = cv2.split(lab_image)
    
    # Normalize LAB channels to [0, 255] (since L is [0, 100] and A/B are [-128, 127])
    a_channel = a_channel + 128  # Shift A from [-128, 127] to [0, 255]
    b_lab_channel = b_lab_channel + 128  # Shift B from [-128, 127] to [0, 255]
    
    # Convert to HSV color space and split into H, S, V channels
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)
    
    # Stack all channels together: RGB (R, G, B), LAB (L, A, B), HSV (H, S, V)
    stacked_image = np.stack([r_channel, g_channel, b_channel,  # RGB (0-255 range)
                              l_channel, a_channel, b_lab_channel,  # LAB (normalized)
                              h_channel, s_channel, v_channel], axis=2)  # HSV (0-255 range)
    return stacked_image


def chip_and_resize(stacked_image, mask, img_name, lbl_name, crop_type, time_of_acquisition, mask_score, chip_size=512,
                     overlap=32, resized_size=224, output_dir=None, threshold=0.8):
    img_h, img_w = stacked_image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    
    assert img_h == mask_h and img_w == mask_w, "Image and mask must have the same dimensions"

    stride = chip_size - overlap  # Define the stride (how much to move in each step)

    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    chip_count = 0
    chip_data = []

    # Assign mask values based on crop type
    if crop_type == 'Maize':
        mask[mask > 0] = 1  # Set all positive pixels to class 1 for Maize
    elif crop_type == 'Soybean':
        mask[mask > 0] = 2  # Set all positive pixels to class 2 for Soybean
    else:
        mask[mask > 0] = 0  # Set all other crop types to class 0

    # Check if there are any positive values in the mask at all
    if not np.any(mask > 0):
        # No positive values found, generate 5 non-overlapping chips from the center of the image
        center_y, center_x = img_h // 2, img_w // 2
        
        # Offsets for 5 non-overlapping chips around the center
        offsets = [
            (-chip_size//2, -chip_size//2), # Center
            (-chip_size//2, 0),             # Right
            (-chip_size//2, chip_size//2),   # Far Right
            (0, -chip_size//2),              # Bottom
            (chip_size//2, -chip_size//2)    # Far Bottom
        ]

        for i, (dy, dx) in enumerate(offsets):
            start_y = max(0, min(img_h - chip_size, center_y + dy))
            start_x = max(0, min(img_w - chip_size, center_x + dx))
            
            # Extract the chip from the stacked image and mask
            img_chip = stacked_image[start_y:start_y + chip_size, start_x:start_x + chip_size]
            mask_chip = mask[start_y:start_y + chip_size, start_x:start_x + chip_size]
            
            # Resize the stacked image and mask to 224x224
            img_resized = cv2.resize(img_chip, (resized_size, resized_size), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_chip, (resized_size, resized_size), interpolation=cv2.INTER_NEAREST)
            
            # Save the stacked image as a .npy file (to preserve the channels) and mask as .png
            img_filename = os.path.join(output_dir, 'images', f'{img_name}_chip_{chip_count:04d}.npy')
            mask_filename = os.path.join(output_dir, 'masks', f'{lbl_name}_chip_{chip_count:04d}.png')
            
            np.save(img_filename, img_resized)  # Save stacked image
            cv2.imwrite(mask_filename, mask_resized)
            
            # Append chip information to the list
            chip_data.append({
                'img_chip_path': str(img_filename).replace(f"{output_dir}/", ""),
                'lbl_chip_path': str(mask_filename).replace(f"{output_dir}/", ""),
                'origin_img': img_name,
                'time_of_acquisition': time_of_acquisition,
                'crop_type': crop_type,
                'original_mask_score': mask_score
            })
            
            chip_count += 1

    else:
        # Positive values found, proceed with normal chipping
        for y in range(0, img_h - chip_size + 1, stride):
            for x in range(0, img_w - chip_size + 1, stride):
                # Extract the chip from the stacked image and mask
                img_chip = stacked_image[y:y + chip_size, x:x + chip_size]
                mask_chip = mask[y:y + chip_size, x:x + chip_size]
                
                # Calculate the percentage of positive pixels in the mask
                positive_pixels = np.count_nonzero(mask_chip)
                total_pixels = mask_chip.size
                positive_ratio = positive_pixels / total_pixels
                
                if positive_ratio >= threshold:
                    # Resize the stacked image and mask to 224x224
                    img_resized = cv2.resize(img_chip, (resized_size, resized_size), interpolation=cv2.INTER_LINEAR)
                    mask_resized = cv2.resize(mask_chip, (resized_size, resized_size), interpolation=cv2.INTER_NEAREST)
                    
                    # Save the stacked image as a .npy file and mask as .png
                    img_filename = os.path.join(output_dir, 'images', f'{img_name}_chip_{chip_count:04d}.npy')
                    mask_filename = os.path.join(output_dir, 'masks', f'{lbl_name}_chip_{chip_count:04d}.png')
                    
                    np.save(img_filename, img_resized)  # Save stacked image
                    cv2.imwrite(mask_filename, mask_resized)
                    
                    # Append chip information to the list
                    chip_data.append({
                        'img_chip_path': str(img_filename).replace(f"{output_dir}/", ""),
                        'lbl_chip_path': str(mask_filename).replace(f"{output_dir}/", ""),
                        'origin_img': img_name,
                        'time_of_acquisition': time_of_acquisition,
                        'crop_type': crop_type,
                        'original_mask_score': mask_score
                    })
                    
                    chip_count += 1
    
    return chip_data

# Function to process the dataframe and generate chips
def process_image_dataframe(df, usage, chip_size=512, overlap=32, resized_size=224, output_dir='output'):
    all_chip_data = []

    # Iterate over each row in the DataFrame with a tqdm notebook progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        img_path = row['img_path'].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
        mask_path = row['mask_path'].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/")
        img_name = row['img_name'].split(".")[0]
        lbl_name = row['mask_name'].split(".")[0]
        crop_type = row['crop_type']
        time_of_acquisition = row['time']
        mask_score = row["mask_score"]
        
        # Load the image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Assuming mask is grayscale
        
        if image is None or mask is None:
            print(f"Error loading image or mask for {img_name}. Skipping...")
            continue
        
        # Generate the stacked image with RGB, LAB, and HSV channels
        stacked_image = generate_stacked_image(image)
        
        # Process the stacked image and mask to generate chips
        chip_data = chip_and_resize(
            stacked_image, mask, img_name, lbl_name, crop_type, time_of_acquisition, mask_score,
            chip_size=chip_size, overlap=overlap, resized_size=resized_size, output_dir=output_dir
        )
        
        # Append the chip data to the overall list
        all_chip_data.extend(chip_data)

    # Create a new DataFrame with the chip information
    chip_df = pd.DataFrame(all_chip_data)
    chip_df.to_csv(os.path.join(output_dir, f"{usage}_chipping_csv.csv"))
    return chip_df

# Function to load the npy file and display the RGB image
def display_rgb_from_npy(npy_file, plot=True, return_rgb=False):
    # Load the stacked image from the npy file
    stacked_image = np.load(npy_file)
    
    # Extract the RGB channels (assuming they are the first three channels)
    r_channel = stacked_image[:, :, 0]
    g_channel = stacked_image[:, :, 1]
    b_channel = stacked_image[:, :, 2]
    
    # Stack the R, G, B channels into an RGB image
    rgb_image = np.stack([r_channel, g_channel, b_channel], axis=2).astype(np.uint8)

    if plot:    
        # Display the RGB image using matplotlib
        plt.imshow(rgb_image)
        plt.axis('off')  # Hide the axes
        plt.show()
    if return_rgb:
        return rgb_image

def display_img_lbl_pair(df, idx, save_path):
    # Load the image and mask paths
    img_path = os.path.join(save_path, df.loc[idx, 'img_chip_path'])
    mask_path = os.path.join(save_path, df.loc[idx, 'lbl_chip_path'])
    crop_type = df.loc[idx, "crop_type"]

    # Load the RGB image using the provided function
    rgb_img = display_rgb_from_npy(img_path, plot=False, return_rgb=True)  # Modified function to return the RGB image

    # Load the mask (label)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Define a custom colormap for the mask where:
    # 0 -> black, 1 -> green, 2 -> blue (You can adjust these colors)
    cmap = mcolors.ListedColormap(['black', 'green', 'blue'])

    # Create a figure with two subplots (side by side)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the RGB image
    ax[0].imshow(rgb_img)
    ax[0].set_title("RGB Image")
    ax[0].axis('off')  # Hide the axes
    
    # Plot the mask with the custom colormap
    # im = ax[1].imshow(mask, cmap=cmap, vmin=0, vmax=2)  # Use vmin and vmax to define the pixel value range
    ax[1].imshow(mask, cmap=cmap, vmin=0, vmax=2)  # Use vmin and vmax to define the pixel value range

    ax[1].set_title(f"Mask (Label) | {crop_type}")
    ax[1].axis('off')  # Hide the axes
    
    # # Add a colorbar for the mask
    # cbar = fig.colorbar(im, ax=ax[1], ticks=[0, 1, 2])
    # cbar.ax.set_yticklabels(['Background (0)', 'Class 1 (1)', 'Class 2 (2)'])  # Custom labels for the colorbar
    
    # Display the plot
    plt.show()


def calculate_npy_dataset_mean_std(dataframe, image_column='img_chip_path',
                                    save_path=None, return_value = False):
    """
    Calculate the mean and standard deviation for each channel in the dataset using pre-saved .npy files.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the paths to the .npy files.
        image_column (str): The column name in the DataFrame that contains the .npy file paths.
    
    Returns:
        (mean, std): Tuple containing the per-channel mean and standard deviation.
    """
    # Initialize lists to store pixel values for each channel
    pixel_values = [[] for _ in range(9)]  # Assuming 9 channels in the stacked images

    # Iterate through the dataset
    for npy_path in tqdm(dataframe[image_column]):
        _path = os.path.join(save_path, npy_path)
        # Load the stacked image (9-channel) from the .npy file
        stacked_image = np.load(_path)  # Shape should be (H, W, 9)

        # Reshape the 9-channel image to (num_pixels, 9)
        flat_image = stacked_image.reshape(-1, 9)

        # Append the pixel values of each channel to the corresponding list
        for i in range(9):
            pixel_values[i].extend(flat_image[:, i])

    # Convert pixel values to numpy arrays for easier manipulation
    pixel_values = [np.array(values) for values in pixel_values]

    # Calculate mean and std for each channel
    mean = np.array([np.mean(values) for values in pixel_values])
    std = np.array([np.std(values) for values in pixel_values])

    print(f"Mean of dataset is:\n{mean}")
    print(f"Std of dataset is:\n{std}")

    if return_value:
        return mean, std

