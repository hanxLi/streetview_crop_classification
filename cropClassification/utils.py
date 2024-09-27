import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import exifread

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
    for crop in crop_calendar:
        if crop["crop_type"].lower() == crop_type.lower():
            if month in crop["planting"]:
                return "planting"
            elif month in crop["growing"]:
                return f"growing"
            elif month in crop["harvesting"]:
                return f"harvesting"
            else:
                return f"Not on Calendar"
            
def show_df_img(df_path, idx):
    df = gpd.read_file(df_path, driver="GEOJSON")
    image = Image.open(df.loc[idx, "save_path"].replace("/home/hanxli/data/", "/Users/steeeve/Documents/csiss/"))
    image = np.array(image.convert("RGB"))
    plt.imshow(image)


def get_image_direction(image_path):
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