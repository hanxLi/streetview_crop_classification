import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import exifread

import pandas as pd
import requests
import time
import os
import google_streetview.api as gsv
from tqdm.notebook import tqdm


crop_calendar = [
    {"crop_type" : "Maize",
    "planting": [4, 5],
    "growing": [6, 7, 8],
    "harvesting": [9, 10, 11]},
    {"crop_type" : "Soybean",
    "planting": [5, 6],
    "growing": [7, 8],
    "harvesting": [9, 10]}]

def check_crop_stage(crop_type, month):
    """
    Check the crop stage based on the crop type and month.

    Parameters
    ----------
    crop_type : str
        The type of crop (e.g., "wheat", "rice").
    month : int
        The current month (represented as an integer, e.g., 1 for January, 12 for December).

    Returns
    -------
    str
        The current crop stage, which could be one of:
        - "planting"
        - "growing"
        - "harvesting"
        - "Not on Calendar"
    """
    for crop in crop_calendar:
        if crop["crop_type"].lower() == crop_type.lower():
            if month in crop["planting"]:
                return "planting"
            elif month in crop["growing"]:
                return "growing"
            elif month in crop["harvesting"]:
                return "harvesting"
            else:
                return "Not on Calendar"
            
def print_common_elements(array1, array2):
    """
    Find and print common elements between two arrays.

    Parameters
    ----------
    array1 : list
        The first list of elements.
    array2 : list
        The second list of elements.

    Returns
    -------
    None
        Prints the common elements if found, otherwise prints a message.
    """
    common_elements = set(array1) & set(array2)
    
    if common_elements:
        print("Common elements:", *common_elements)
    else:
        print("No matching elements")


def plot_bar(df, title):
    """
    Plot a bar graph showing the distribution of crop types in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing crop type information in the 'crop_type' column.
    title : str
        The title of the plot.
    
    Returns
    -------
    None
        Displays the bar plot.
    """
    # Handle the case where the DataFrame is empty
    if df.empty or 'crop_type' not in df.columns:
        print("DataFrame is empty or 'crop_type' column is missing.")
        return
    
    # Calculate unique crop types and their counts
    temp = np.unique(df.crop_type, return_counts=True)
    total_count = np.sum(temp[1])
    percentage = [count / total_count * 100 for count in temp[1]]

    # Plot bar chart
    fig = plt.figure(figsize=(10,5))
    graph = plt.bar(temp[0], temp[1])
    
    # Annotate the bars with percentage labels
    for p, perc in zip(graph, percentage):
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x + width / 2,
                 y + height * 1.01,
                 f'{round(perc, 2)}%',
                 ha='center',
                 weight='bold')

    # Add labels and title
    plt.ylabel('Count')
    plt.xticks(rotation=25)
    plt.title(title)

    # Show the plot
    plt.show()


def show_df_img(df_path, idx):
    """
    Display an image from a GeoJSON file based on the row index.

    Parameters
    ----------
    df_path : str
        The file path to the GeoJSON file.
    idx : int
        The index of the row in the DataFrame that corresponds to the image.

    Returns
    -------
    None
        The function displays the image using matplotlib.
    """
    df = gpd.read_file(df_path, driver="GEOJSON")
    image = Image.open(df.loc[idx, "save_path"].replace("/home/hanxli/data/", 
                                                        "/Users/steeeve/Documents/csiss/"))
    image = np.array(image.convert("RGB"))
    plt.imshow(image)



def get_image_direction(image_path):
    """
    Extract the GPS direction from the metadata of an image file.

    Parameters
    ----------
    image_path : str
        The file path to the image.

    Returns
    -------
    float or str
        The GPS direction in degrees if available, or a message indicating that the direction is not available.
    """
    # Open the image file
    with open(image_path, 'rb') as image_file:
        # Use exifread to extract Exif data
        tags = exifread.process_file(image_file)

        # Extract GPS direction if available
        direction = tags.get('GPS GPSImgDirection')
        if direction:
            # Convert direction to float value
            direction_value = float(direction.values[0].num) / float(direction.values[0].den)
            return direction_value
        else:
            return "Direction not available in the metadata."


def get_single_gsv(api_key, pano_id=None, location=None, download_path=None):
    """
    Fetch a Google Street View image by pano_id or location and download it.

    Parameters
    ----------
    api_key : str
        The API key for authenticating with the Google Street View API.
    pano_id : str, optional
        The panorama ID for a specific Street View image. Default is None.
    location : str, optional
        The geolocation (e.g., address or lat/lng) for fetching a Street View image. Default is None.
    download_path : str, optional
        The path where the image will be saved. If not provided, a folder named 'gsv_img' will be created in the current working directory.
    
    Raises
    ------
    ValueError
        If both location and pano_id are None.
    
    Returns
    -------
    None
        The function downloads the image to the specified or default location.
    """

    # Raise error if both pano_id and location are not provided
    if location is None and pano_id is None:
        raise ValueError("Both location and pano_id cannot be None. Please provide at least one geospatial identifier.")

    # Prepare the parameters for the Google Street View API request
    params = [{
        'size': '640x640',
        'return_error_code': "true",
        'key': api_key
    }]

    # Set pano_id or location in the parameters
    if pano_id is not None:
        params[0]['pano'] = pano_id
    else:
        params[0]['location'] = location

    results = gsv.results(params)

    # If no download_path is provided, create the 'gsv_img' folder in the current working directory
    if download_path is None:
        download_path = os.path.join(os.getcwd(), 'gsv_img')
        os.makedirs(download_path, exist_ok=True)

    # Download the image using the results object
    results.download_links(download_path)


def check_gsv_status(df, api_key, timeout=10):
    """
    Check the status of Google Street View images using the pano_id from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the 'imageIdKey' column with Google Street View pano IDs.
    api_key : str
        The API key for authenticating with the Google Street View API.
    
    Returns
    -------
    df : pandas.DataFrame
        The updated DataFrame with a new column 'gsv_current_status' showing the status of each pano ID.
    """
    gsv_meta_base = "https://maps.googleapis.com/maps/api/streetview/metadata?"

    # Iterate through DataFrame with progress bar
    for i in tqdm(range(len(df)), desc="Checking GSV Status"):
        gsv_pano_id = df.loc[i, 'imageIdKey']

        try:
            response = requests.get(gsv_meta_base, params={
                'pano': gsv_pano_id,
                'key': api_key},
                timeout=timeout)

            # Check if the request was successful
            response.raise_for_status()

            # Extract the status from the response JSON
            status = response.json().get("status", "Unknown")

            # Assign status to the DataFrame
            df.loc[i, 'gsv_current_status'] = status

        except requests.exceptions.RequestException as e:
            # If there's an error in the request, log it as an error in the status column
            df.loc[i, 'gsv_current_status'] = f"Error: {e}"
        except ValueError:
            # If there's an issue with JSON decoding
            df.loc[i, 'gsv_current_status'] = "Invalid JSON response"

    return df

def download_gsv(df, path, api_key, timeout=10, return_df=True):
    """
    Download Google Street View images for given pano IDs from a DataFrame and track metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing image metadata including pano IDs.
    path : str
        The directory path where the images will be saved. If the directory doesn't exist, it will be created.
    api_key : str
        The API key for authenticating with the Google Street View API.
    timeout : int, optional
        The maximum time (in seconds) to wait for a response from the API before timing out. Default is 10 seconds.

    Returns
    -------
    pandas.DataFrame
        A DataFrame tracking the downloaded images and their corresponding metadata.
    """
    gsv_url_base = "https://maps.googleapis.com/maps/api/streetview?"
    pic_params = {
        'size': "640x640",
        'pano': None,
        'key': api_key
    }

    img_tracker = []

    # Ensure the directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Progress bar for tracking the download progress
    for i in tqdm(range(len(df)), desc="Downloading GSV images"):
        try:
            pic_name = f"img_{df.loc[i, 'unique_id']}_{df.loc[i, 'id']}.jpg"
            pic_params["pano"] = df.loc[i, "imageIdKey"]
            save_path = os.path.join(path, pic_name)

            # Download the image if it does not already exist
            if not os.path.isfile(save_path):
                gsv_response = requests.get(gsv_url_base, params=pic_params, timeout=timeout)
                gsv_response.raise_for_status()  # Raise an exception for HTTP errors

                with open(save_path, 'wb') as f:
                    f.write(gsv_response.content)
                
                # Optional sleep to prevent potential rate-limiting issues (could be adjusted or removed)
                time.sleep(0.001)

            # Extract latitude and longitude from the 'imgLoc' field
            latlon = df.loc[i, "imgLoc"].split(",")
            lat = latlon[1].replace("lng:", "").strip()
            lon = latlon[0].replace("lat:", "").strip()

            # Create a dictionary with metadata
            temp_dict = {
                'img_name': pic_name,
                'date': df.loc[i, "svImgDate"],
                'crop_type': df.loc[i, "cropType"],
                'lat': lat,
                'lon': lon,
                'save_path': save_path
            }

            img_tracker.append(temp_dict)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image {pic_name}: {e}")
        except (IndexError, ValueError) as e:
            print(f"Error parsing lat/lon for image {pic_name}: {e}")
        except Exception as e:
            print(f"Unexpected error with image {pic_name}: {e}")
    temp_df = pd.DataFrame(img_tracker)

    temp_df.to_csv(os.path.join(path, "ESA_GSV_IMG.csv"), sep=',', index=False, header=True)
    if return_df:
        return temp_df