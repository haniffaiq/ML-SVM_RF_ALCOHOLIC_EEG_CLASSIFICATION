import cv2
import os
import numpy as np
from tqdm import tqdm

# Set the path for the input folder containing grayscale images
input_path = 'Data_Normal/'

# Set the path for the output folder where the enhanced images will be saved
output_path = 'Data_Enhance/'

# Define the new range for contrast stretching
new_min_val = 64
new_max_val = 256

# Loop through each file in the input folder
for filename in  tqdm(os.listdir(input_path), desc='Loading Export Enhancement Images'):

    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):

        # Load the image in grayscale
        img = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)

        # Calculate minimum and maximum pixel values
        min_val, max_val, _, _ = cv2.minMaxLoc(img)

        # Calculate new pixel values
        new_img = np.uint8((img - min_val) * (new_max_val - new_min_val) / (max_val - min_val) + new_min_val)

        # Save the enhanced image to the output folder
        cv2.imwrite(os.path.join(output_path, filename), new_img)

