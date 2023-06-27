import os
import time
import shutil
from PIL import Image
from nsfw_model import NSFWDetector

# Create NSFW detector instance
nsfw_detector = NSFWDetector()


# Function to check the range for NSFW probability and return corresponding folder
def get_folder_name(nsfw_probability):
    if nsfw_probability is None:
        return 'None'
    elif nsfw_probability < 0.2:
        return '<0.2'
    elif nsfw_probability < 0.4:
        return '<0.4'
    elif nsfw_probability < 0.6:
        return '<0.6'
    elif nsfw_probability < 0.8:
        return '<0.8'
    else:
        return '>=0.8'


# Function to check if an image is NSFW
def is_nsfw(image_path):
    def is_nsfw(image_path):
        # Load image and resize to maximum of 512 pixels
        img = Image.open(image_path)
        img.thumbnail((512, 512))

        # Check NSFW probability using the NSFW detector
        nsfw_probability = nsfw_detector.predict(img)
        return nsfw_probability


# Define input and output directories
input_folder = "E:/Stable-Diffusion/stable-diffusion-webui/horde/2023-06-09"
output_folder = "E:/Stable-Diffusion/stable-diffusion-webui/horde/NSFW"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Check if the file is a valid image
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        print(f"Analyzing image: {filename}")

        # Check if the image is NSFW
        nsfw_probability = is_nsfw(file_path)
        print(f"NSFW probability: {nsfw_probability}")

        # Determine the appropriate folder based on NSFW probability
        folder_name = get_folder_name(nsfw_probability)
        print(f"Detected NSFW image: {filename} -> Moving to folder: {folder_name}")

        # Move image to the output folder
        output_folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        shutil.move(file_path, os.path.join(output_folder_path, filename))
        print(f"Moved image to {output_folder_path}/{filename}")
        time.sleep(5)
    else:
        print(f"Skipping non-image file: {filename}")

print("Image analysis and sorting complete.")
