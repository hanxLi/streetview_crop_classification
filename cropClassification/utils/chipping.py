import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pywt
from tqdm.notebook import tqdm

def update_masks(df, backup_masks=False):
    """
    Modify mask files based on crop_type:
      - If 'Maize', keep valid region as 1, others as 0.
      - If 'Soybean', convert valid region (1) to 2, others as 0.
      - If 'Other', set all pixels to 0 (no valid region).

    Args:
    -----
    df (pd.DataFrame): DataFrame with crop types and mask paths.
    backup_masks (bool): Whether to save a backup of the original masks (default: False).
    """
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Masks"):
        mask_path = row['mask_path']
        crop_type = row['crop_type']

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}. Skipping...")
            continue

        try:
            # Load the mask as a NumPy array
            mask = np.array(Image.open(mask_path))

            # Validate that the mask contains only 1s for valid regions and 0s elsewhere
            mask = np.where(mask == 1, 1, 0)

            # Backup the original mask if required
            if backup_masks:
                backup_path = mask_path.replace(".png", "_backup.png")
                Image.fromarray(mask).save(backup_path)

            # Process the mask based on crop_type
            if crop_type == 'Maize':
                # Valid region remains as 1, others as 0 (already correct)
                pass

            elif crop_type == 'Soybean':
                # Convert valid region (1) to 2, others remain 0
                mask = np.where(mask == 1, 2, 0)

            elif crop_type == 'Other':
                # Set all pixels to 0
                mask.fill(0)

            # Ensure the mask is dtype uint8 for PIL compatibility
            mask = mask.astype(np.uint8)

            # Save the modified mask
            Image.fromarray(mask).save(mask_path)

        except Exception as e:
            print(f"Error processing mask at {mask_path}: {e}")




def compute_wavelet_features(image, wavelet='db1'):
    """
    Compute Wavelet Transform features for a grayscale image.

    Args:
        image (np.ndarray): Grayscale input image (H, W).
        wavelet (str): Type of wavelet to use (e.g., 'db1').

    Returns:
        np.ndarray: Approximation coefficients resized to the original shape (H, W).
    """
    # Ensure the image is in float format
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Perform 2D discrete wavelet transform
    coeffs2 = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs2  # Approximation and detail coefficients

    # Resize the approximation coefficients back to the original image size
    wavelet_features = cv2.resize(cA, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize the wavelet features to [0, 1] range
    wavelet_features = (wavelet_features - wavelet_features.min()) / \
                       (wavelet_features.max() - wavelet_features.min() + 1e-6)

    return wavelet_features

def generate_stacked_image(image):
    """
    Convert an RGB image to a 10-channel image by stacking RGB, LAB, HSV, and Wavelet channels.
    
    Args:
        image (np.ndarray): Input RGB image (H, W, 3).
    
    Returns:
        np.ndarray: 10-channel image (H, W, 10).
    """
    # Split the image into RGB channels
    b_channel, g_channel, r_channel = cv2.split(image)  # OpenCV loads as BGR by default
    
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_lab_channel = cv2.split(lab_image)

    # Shift LAB channels to the range [0, 255]
    a_channel += 128
    b_lab_channel += 128

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    # Compute Wavelet features as the 10th channel (from grayscale image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    wavelet_channel = compute_wavelet_features(gray_image)  # (H, W)

    # print(f"RGB shapes: {r_channel.shape}, {g_channel.shape}, {b_channel.shape}")
    # print(f"LAB shapes: {l_channel.shape}, {a_channel.shape}, {b_lab_channel.shape}")
    # print(f"HSV shapes: {h_channel.shape}, {s_channel.shape}, {v_channel.shape}")
    # print(f"Wavelet shape: {wavelet_channel.shape}")

    # Stack all channels together into a 10-channel image
    stacked_image = np.stack([
        r_channel, g_channel, b_channel,  # RGB
        l_channel, a_channel, b_lab_channel,  # LAB
        h_channel, s_channel, v_channel,  # HSV
        wavelet_channel  # Wavelet features
    ], axis=2)  # (H, W, 10)

    return stacked_image


# def chip_and_resize(stacked_image, mask, img_name, lbl_name, crop_type, time_of_acquisition, mask_score,
#                     chip_size=512, overlap=32, resized_size=224, output_dir=None, threshold=0.5,
#                     other_class_ratio=0.1):
#     """
#     Generate and resize chips from the stacked image and mask with dominant class logic for mixed chips.
    
#     Args:
#         other_class_ratio (float): Max ratio of 'other' class chips to be included (0 to 1).
#     """
#     img_h, img_w = stacked_image.shape[:2]
#     stride = chip_size - overlap

#     if output_dir is None:
#         output_dir = os.getcwd()

#     os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

#     chip_data = []
#     chip_count = 0
#     other_chip_count = 0
#     target_chip_count = 0

#     for y in range(0, img_h - chip_size + 1, stride):
#         for x in range(0, img_w - chip_size + 1, stride):
#             img_chip = stacked_image[y:y + chip_size, x:x + chip_size]
#             mask_chip = mask[y:y + chip_size, x:x + chip_size]

#             # Ensure only valid mask values [0, 1, 2]. Reset others to 0.
#             mask_chip = np.where(np.isin(mask_chip, [0, 1, 2]), mask_chip, 0)

#             # Handle mixed chips (both 1 and 2 present)
#             unique_values = np.unique(mask_chip) 
#             if 1 in unique_values and 2 in unique_values:
#                 # Determine the dominant value (most frequent between 1 and 2)
#                 counts = np.bincount(mask_chip.flatten(), minlength=3)  # Counts for [0, 1, 2]
#                 dominant_value = np.argmax(counts[1:]) + 1  # Ignore background (0)
#                 mask_chip = np.where(mask_chip > 0, dominant_value, 0)

#             # Calculate the positive ratio (non-background pixels)
#             positive_ratio = np.count_nonzero(mask_chip) / mask_chip.size

#             # Track target chips and other chips
#             if positive_ratio >= threshold:
#                 target_chip_count += 1
#             elif other_chip_count >= other_class_ratio * target_chip_count:
#                 continue  # Skip if too many "other" chips
#             else:
#                 other_chip_count += 1

#             # Resize the chip and mask to the desired size
#             img_resized = cv2.resize(img_chip, (resized_size, resized_size), interpolation=cv2.INTER_LINEAR)
#             mask_resized = cv2.resize(mask_chip, (resized_size, resized_size), interpolation=cv2.INTER_NEAREST)

#             # Ensure the resized mask contains only valid labels [0, 1, 2]
#             mask_resized = np.round(mask_resized).astype(np.uint8)
#             mask_resized = np.where(np.isin(mask_resized, [0, 1, 2]), mask_resized, 0)

#             # Save the image and mask chips
#             img_filename = os.path.join(output_dir, 'images', f'{img_name}_chip_{chip_count:04d}.npy')
#             mask_filename = os.path.join(output_dir, 'masks', f'{lbl_name}_chip_{chip_count:04d}.png')

#             np.save(img_filename, img_resized)
#             cv2.imwrite(mask_filename, mask_resized)

#             # Append chip information to the list
#             chip_data.append({
#                 'img_chip_path': str(img_filename).replace(f"{output_dir}/", ""),
#                 'lbl_chip_path': str(mask_filename).replace(f"{output_dir}/", ""),
#                 'origin_img': img_name,
#                 'time_of_acquisition': time_of_acquisition,
#                 'crop_type': crop_type if positive_ratio >= threshold else 'Other',  # Assign 'Other' class
#                 'original_mask_score': mask_score
#             })

#             chip_count += 1

#     return chip_data

def chip_and_resize(stacked_image, mask, img_name, lbl_name, crop_type, time_of_acquisition, mask_score,
                    chip_size=512, overlap=32, resized_size=None, output_dir=None, threshold=0.5,
                    other_class_ratio=0.3, method='mixed'):
    """
    Generate and resize chips from the stacked image and mask with optional mixed or pure logic.

    Args:
        stacked_image (np.array): Input stacked image.
        mask (np.array): Corresponding mask.
        img_name (str): Name of the input image file.
        lbl_name (str): Name of the label file.
        crop_type (str): Crop type for metadata.
        time_of_acquisition (str): Time of acquisition metadata.
        mask_score (float): Score associated with the original mask.
        chip_size (int): Size of the chips (default is 512).
        overlap (int): Overlap between chips (default is 32).
        resized_size (int or None): Size to resize the chips (default is None, no resizing).
        output_dir (str or None): Directory to save output chips (default is current directory).
        threshold (float): Minimum positive ratio for target chips (default is 0.5).
        other_class_ratio (float): Max ratio of 'other' class chips to include (default is 0.3).
        method (str): 'mixed' for dominant-class logic, 'pure' for pure chips only.

    Returns:
        list: Metadata of generated chips.
    """
    img_h, img_w = stacked_image.shape[:2]
    stride = chip_size - overlap

    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    chip_data = []
    chip_count = 0
    target_chip_count = 0
    other_chip_count_top = 0  # Count of "Other" chips from the top half
    other_chip_count_bottom = 0  # Count of "Other" chips from the bottom half
    other_source_images = set()  # Track images contributing "Other" chips

    for y in range(0, img_h - chip_size + 1, stride):
        for x in range(0, img_w - chip_size + 1, stride):
            img_chip = stacked_image[y:y + chip_size, x:x + chip_size]
            mask_chip = mask[y:y + chip_size, x:x + chip_size]

            # Ensure only valid labels (0, 1, 2) are present in the mask chip
            mask_chip = np.where(np.isin(mask_chip, [0, 1, 2]), mask_chip, 0)

            if method == 'mixed':
                # Handle mixed chips (both 1 and 2 present)
                unique_values = np.unique(mask_chip)
                if 1 in unique_values and 2 in unique_values:
                    print(f"Mixed chip detected: {img_name}_chip_{chip_count:04d}")
                    counts = np.bincount(mask_chip.flatten(), minlength=3)  # Counts for [0, 1, 2]
                    dominant_value = np.argmax(counts[1:]) + 1  # Ignore background (0)
                    mask_chip = np.where(mask_chip > 0, dominant_value, 0)

                positive_ratio = np.count_nonzero(mask_chip) / mask_chip.size
                if positive_ratio < threshold:
                    continue  # Skip chips with low positive ratio

            elif method == 'pure':
                positive_ratio = np.count_nonzero(mask_chip) / mask_chip.size

                if positive_ratio == 1.0:  # Pure target chip (Maize or Soybean)
                    target_chip_count += 1  # Increment target chip count
                elif positive_ratio == 0.0:  # Pure "Other" chip
                    other_source_images.add(img_name)  # Track the source image for diversity

                    max_other_chips = float('inf') if target_chip_count == 0 else int(other_class_ratio * target_chip_count / 2)

                    if y < img_h // 2:
                        if other_chip_count_top >= max_other_chips:
                            continue  # Skip if too many "Other" chips from the top
                        other_chip_count_top += 1
                    else:
                        if other_chip_count_bottom >= max_other_chips:
                            continue  # Skip if too many "Other" chips from the bottom
                        other_chip_count_bottom += 1
                else:
                    continue  # Skip chips that aren't purely target or "Other"

            if resized_size is not None:
                img_chip = cv2.resize(img_chip, (resized_size, resized_size), interpolation=cv2.INTER_LINEAR)
                mask_chip = cv2.resize(mask_chip, (resized_size, resized_size), interpolation=cv2.INTER_NEAREST)

            # Save the image and mask chips
            img_filename = os.path.join(output_dir, 'images', f'{img_name}_chip_{chip_count:04d}.npy')
            mask_filename = os.path.join(output_dir, 'masks', f'{lbl_name}_chip_{chip_count:04d}.png')

            np.save(img_filename, img_chip)
            cv2.imwrite(mask_filename, mask_chip)

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
# def process_image_dataframe(df, usage, chip_size=512, overlap=32, resized_size=224, output_dir='output', threshold=0.8):
#     all_chip_data = []

#     # Iterate over each row in the DataFrame with a tqdm notebook progress bar
#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
#         img_path = row['img_path']
#         mask_path = row['mask_path']
#         img_name = row['img_name'].split(".")[0]
#         lbl_name = row['mask_name'].split(".")[0]
#         crop_type = row['crop_type']
#         time_of_acquisition = row['time']
#         mask_score = row["mask_score"]
        
#         # Load the image and mask
#         image = cv2.imread(img_path)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Assuming mask is grayscale
        
#         if image is None or mask is None:
#             print(f"Error loading image or mask for {img_name}. Skipping...")
#             continue
        
#         # Generate the stacked image with RGB, LAB, and HSV channels
#         stacked_image = generate_stacked_image(image)
        
#         # Process the stacked image and mask to generate chips
#         chip_data = chip_and_resize(
#             stacked_image, mask, img_name, lbl_name, crop_type, time_of_acquisition, mask_score,
#             chip_size=chip_size, overlap=overlap, resized_size=resized_size, output_dir=output_dir,threshold=threshold
#         )
        
#         # Append the chip data to the overall list
#         all_chip_data.extend(chip_data)

#     # Create a new DataFrame with the chip information
#     chip_df = pd.DataFrame(all_chip_data)
#     chip_df.to_csv(os.path.join(output_dir, f"{usage}_chipping_csv.csv"))
#     return chip_df

def process_image_dataframe(df, usage, chip_size=512, overlap=32, resized_size=None, output_dir='output',
                            threshold=0.8, method='mixed', other_class_ratio=0.3):
    """
    Process a DataFrame of images and masks to generate chips using the chip_and_resize function.
    
    Args:
        df (pd.DataFrame): DataFrame containing image and mask paths, metadata, etc.
        usage (str): Usage label for output (e.g., 'train', 'val', or 'test').
        chip_size (int): Size of the chips to extract (default is 512).
        overlap (int): Overlap between chips (default is 32).
        resized_size (int or None): Size to resize chips, or None to skip resizing (default is None).
        output_dir (str): Directory to save chip data (default is 'output').
        threshold (float): Minimum positive ratio for target chips (default is 0.8).
        method (str): 'mixed' for dominant-class logic, 'pure' for pure chips only (default is 'mixed').
        other_class_ratio (float): Max ratio of 'other' class chips to include (default is 0.3).
    
    Returns:
        pd.DataFrame: DataFrame with metadata about generated chips.
    """
    all_chip_data = []

    # Iterate over each row in the DataFrame with a progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        img_path = row['img_path']
        mask_path = row['mask_path']
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
            chip_size=chip_size, overlap=overlap, resized_size=resized_size, output_dir=output_dir,
            threshold=threshold, method=method, other_class_ratio=other_class_ratio
        )

        # Append the chip data to the overall list
        all_chip_data.extend(chip_data)

    # Create a new DataFrame with the chip information
    chip_df = pd.DataFrame(all_chip_data)
    chip_df.to_csv(os.path.join(output_dir, f"{usage}_chipping_csv.csv"), index=False)
    print("Total number of chips ", len(chip_df))
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
                                   save_path=None, num_channels=10):
    """
    Calculate the mean and standard deviation for each channel in the dataset incrementally
    to reduce memory usage.
    """
    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sum_sq = np.zeros(num_channels, dtype=np.float64)
    num_pixels = 0

    for npy_path in tqdm(dataframe[image_column], desc="Computing Mean/Std"):
        _path = os.path.join(save_path, npy_path)

        # Load the stacked image
        stacked_image = np.load(_path)

        # Check number of channels
        assert stacked_image.shape[-1] == num_channels, \
            f"Expected {num_channels} channels, got {stacked_image.shape[-1]} in {_path}"

        # Reshape to (num_pixels, num_channels)
        flat_image = stacked_image.reshape(-1, num_channels)

        # Incrementally compute sums and sum of squares
        channel_sum += np.sum(flat_image, axis=0)
        channel_sum_sq += np.sum(flat_image**2, axis=0)
        num_pixels += flat_image.shape[0]

    # Compute mean and std
    mean = channel_sum / num_pixels
    std = np.sqrt((channel_sum_sq / num_pixels) - (mean**2))

    print(f"Mean of dataset: {mean}")
    print(f"Std of dataset: {std}")

    return mean, std


# def calculate_npy_dataset_mean_std(dataframe, image_column='img_chip_path',
#                                    save_path=None, num_channels=10, return_value=False):
#     """
#     Calculate the mean and standard deviation for each channel in the dataset using pre-saved .npy files.
    
#     Args:
#         dataframe (pd.DataFrame): DataFrame containing the paths to the .npy files.
#         image_column (str): The column name in the DataFrame that contains the .npy file paths.
#         save_path (str): Directory path where the .npy files are located.
#         num_channels (int): Number of channels in the stacked images.
#         return_value (bool): Whether to return the computed mean and std values.

#     Returns:
#         (mean, std): Tuple containing the per-channel mean and standard deviation.
#     """
#     # Initialize lists to store pixel values for each channel
#     pixel_values = [[] for _ in range(num_channels)]  # Adjusted based on num_channels

#     # Iterate through the dataset
#     for npy_path in tqdm(dataframe[image_column], desc="Computing Mean/Std"):
#         _path = os.path.join(save_path, npy_path)

#         # Load the stacked image from the .npy file
#         stacked_image = np.load(_path)  # Shape should be (H, W, num_channels)

#         # Check if the image has the expected number of channels
#         assert stacked_image.shape[-1] == num_channels, \
#             f"Expected {num_channels} channels, but got {stacked_image.shape[-1]} in {_path}"

#         # Reshape the image to (num_pixels, num_channels)
#         flat_image = stacked_image.reshape(-1, num_channels)

#         # Append the pixel values of each channel to the corresponding list
#         for i in range(num_channels):
#             pixel_values[i].extend(flat_image[:, i])

#     # Convert pixel values to numpy arrays for easier manipulation
#     pixel_values = [np.array(values) for values in pixel_values]

#     # Calculate mean and std for each channel
#     mean = np.array([np.mean(values) for values in pixel_values])
#     std = np.array([np.std(values) for values in pixel_values])

#     print(f"Mean of dataset is:\n{mean}")
#     print(f"Std of dataset is:\n{std}")

#     if return_value:
#         return mean, std


def calculate_classwise_mean_std(dataframe, image_column='img_chip_path', 
                                 mask_column='lbl_chip_path', root_dir='', 
                                 num_classes=3, num_channels=10, return_value=False):
    """
    Calculate class-wise mean and standard deviation for each channel in the dataset.

    Args:
        dataframe (pd.DataFrame): DataFrame containing paths to .npy images and masks.
        image_column (str): Column name with .npy file paths.
        mask_column (str): Column name with mask paths.
        root_dir (str): Root directory containing the images and masks.
        num_classes (int): Number of classes in the masks.
        num_channels (int): Number of channels in the images.
        return_value (bool): Whether to return the class-wise mean and std.

    Returns:
        classwise_norm (dict): Dictionary with class-wise mean and std per channel.
                               Format: {class_id: {'mean': [...], 'std': [...]}}

    """
    # Initialize dictionaries to store pixel values per class
    pixel_values = {cls: [[] for _ in range(num_channels)] for cls in range(num_classes)}

    # Iterate through the dataset
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        # Load the image and mask paths
        img_path = os.path.join(root_dir, row[image_column])
        mask_path = os.path.join(root_dir, row[mask_column])

        # Load the image and validate its shape
        try:
            image = np.load(img_path)  # Load the .npy image
            assert image.shape[-1] == num_channels, f"Image {img_path} has {image.shape[-1]} channels, expected {num_channels}"
        except (IOError, AssertionError) as e:
            print(f"Skipping {img_path} due to error: {e}")
            continue  # Skip this image if loading or shape fails

        # Load the mask and validate it
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load the mask as grayscale
        if mask is None:
            print(f"Skipping {mask_path} as mask could not be loaded.")
            continue

        # Flatten the image and mask for easier manipulation
        flat_image = image.reshape(-1, num_channels)  # (num_pixels, num_channels)
        flat_mask = mask.flatten()  # (num_pixels,)

        # Collect pixel values per class
        for cls in range(num_classes):
            class_indices = np.where(flat_mask == cls)[0]  # Indices of pixels belonging to this class
            for channel in range(num_channels):
                # Append pixel values for this class and channel
                pixel_values[cls][channel].extend(flat_image[class_indices, channel])

    # Calculate mean and std for each class and channel
    classwise_norm = {}
    for cls in range(num_classes):
        class_mean = np.array([np.mean(pixel_values[cls][ch]) for ch in range(num_channels)])
        class_std = np.array([np.std(pixel_values[cls][ch]) for ch in range(num_channels)])
        classwise_norm[cls] = {'mean': class_mean.tolist(), 'std': class_std.tolist()}

        print(f"Class {cls} - Mean: {class_mean}")
        print(f"Class {cls} - Std: {class_std}")

    if return_value:
        return classwise_norm

def compute_pixel_class_weights(df_train, df_val, label_column='lbl_chip_path', label_dir=''):
    """
    Compute pixel-wise class weights from the combined training and validation data.

    Args:
        df_train (pd.DataFrame): DataFrame containing the paths to training label chips.
        df_val (pd.DataFrame): DataFrame containing the paths to validation label chips.
        label_column (str): The column in the DataFrame containing paths to the label chips.
        label_dir (str): Root directory where 'training' and 'validation' folders are located.

    Returns:
        np.ndarray: Class weights including background (class 0).
    """

    # Initialize an empty dictionary to store pixel counts dynamically
    class_pixel_counts = {}

    def update_pixel_counts(label_path, usage='train'):
        """
        Update the class pixel counts for each label chip.
        """
        folder = "training" if usage == "train" else "validation"
        full_path = os.path.join(label_dir, folder, label_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        # Load the label chip and get unique pixel counts
        label_chip = np.array(Image.open(full_path))
        unique, counts = np.unique(label_chip, return_counts=True)

        # Update pixel counts for all classes (dynamically handle new classes)
        for cls, cnt in zip(unique, counts):
            if cls not in class_pixel_counts:
                class_pixel_counts[cls] = 0  # Initialize if class not seen before
            class_pixel_counts[cls] += cnt

    # Process training and validation datasets
    for label_path in tqdm(df_train[label_column], desc="Training Labels"):
        update_pixel_counts(label_path, usage='train')

    for label_path in tqdm(df_val[label_column], desc="Validation Labels"):
        update_pixel_counts(label_path, usage='val')

    # Define expected classes in order
    classes = [0, 1, 2]

    # Convert pixel counts to a NumPy array in the correct class order
    pixel_counts = np.array([class_pixel_counts.get(cls, 0) for cls in classes])

    # Compute class weights using inverse frequency (or try log scaling)
    class_weights = 1.0 / np.sqrt(pixel_counts)  # Inverse square root frequency
    class_weights /= class_weights.sum()  # Normalize to sum to 1

    print("Class Weights (NumPy):", class_weights)
    print("Class Pixel Counts:", {cls: class_pixel_counts.get(cls, 0) for cls in classes})

    return class_weights

