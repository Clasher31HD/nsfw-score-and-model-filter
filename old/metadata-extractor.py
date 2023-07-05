from PIL import Image
import os

def print_image_metadata(image_path):
    try:
        image = Image.open(image_path)
        metadata = image.info
        print("Image:", os.path.basename(image_path))
        print("Metadata:", metadata)
        print("------------------------")
    except Exception as e:
        print("Error reading image:", os.path.basename(image_path))
        print("Error message:", str(e))
        print("------------------------")

# Example usage
input_folder_path = "E:/Stable-Diffusion/stable-diffusion-webui/horde/2023-06-22"

for filename in os.listdir(input_folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(input_folder_path, filename)
        print_image_metadata(image_path)
