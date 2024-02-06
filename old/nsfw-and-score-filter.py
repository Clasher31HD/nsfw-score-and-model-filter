import os
import time
import shutil
from PIL import Image
import numpy as np
import opennsfw2 as n2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image


# Function to check the range for NSFW probability and return corresponding folder
def get_nsfw_folder_name(nsfw_probability):
    if nsfw_probability is None:
        return 'None'
    elif nsfw_probability < 0.2:
        return '0.0-0.2'
    elif nsfw_probability < 0.4:
        return '0.2-0.4'
    elif nsfw_probability < 0.6:
        return '0.4-0.6'
    elif nsfw_probability < 0.8:
        return '0.6-0.8'
    elif nsfw_probability < 0.9:
        return '0.8-0.9'
    elif nsfw_probability < 0.95:
        return '0.9-0.95'
    elif nsfw_probability < 0.99:
        return '0.95-0.99'
    else:
        return '0.99-1.0'


def get_score_folder_name(score):
    if score is None:
        return 'None'
    elif score < 0.2:
        return '0.2'
    elif score < 0.4:
        return '0.4'
    elif score < 0.6:
        return '0.6'
    elif score < 0.8:
        return '0.8'
    else:
        return '1.0'


# Function to check if an image is NSFW
def is_nsfw(image_path):
    # Load image and resize to maximum of 512 pixels
    img = Image.open(image_path)
    img.thumbnail((512, 512))

    # Check NSFW probability using the NSFW detector
    nsfw_probability = n2.predict_image(image_path)
    return nsfw_probability


def get_image_score(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Use the pre-trained model to predict the image class probabilities
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Return the predicted class index and corresponding score
    return predicted_class, predictions[0][predicted_class]


# Function to validate the mode input
def validate_mode_input(input_str):
    valid_modes = ['1', '2', '3']
    return input_str in valid_modes


mode = input("Select Mode (1 = NSFW, 2 = Score, 3 = Combined): ")

# Validate the mode input
while not validate_mode_input(mode):
    print("Invalid mode input. Please enter a value between 1 and 3.")
    mode = input("Select Mode (1 = NSFW, 2 = Score, 3 = Combined): ")

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Define input and output directories
input_folder = "C:/Users/I539356/Downloads/www.freepik.com/All"
output_folder = "C:/Users/I539356/Downloads/Filtered"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Count the total number of images in the input folder
total_images = len([filename for filename in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, filename))])

# Iterate over all files in the input folder
for idx, filename in enumerate(os.listdir(input_folder)):
    file_path = os.path.join(input_folder, filename)

    # Check if the file is a valid image
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        print(f"\nAnalyzing image {idx+1}/{total_images}\n{filename}")

        if mode == "1":
            # Check if the image is NSFW
            nsfw_probability = is_nsfw(file_path)
            print(f"NSFW probability: {nsfw_probability}")
            nsfw_folder_name = get_nsfw_folder_name(nsfw_probability)
            output_folder_path = os.path.join(output_folder, nsfw_folder_name)
        elif mode == "2":
            # Get the score and index for the input image
            class_index, score = get_image_score(file_path)
            print(f"Score: {score}, Class: {class_index}")
            score_folder_name = get_score_folder_name(score)
            output_folder_path = os.path.join(output_folder, score_folder_name)
        elif mode == "3":
            # Get the nsfw, score and index for the input image
            nsfw_probability = is_nsfw(file_path)
            print(f"NSFW probability: {nsfw_probability}")
            nsfw_folder_name = get_nsfw_folder_name(nsfw_probability)
            class_index, score = get_image_score(file_path)
            print(f"Score: {score}, Class: {class_index}")
            score_folder_name = get_score_folder_name(score)
            output_folder_path = os.path.join(output_folder, nsfw_folder_name, score_folder_name)

        os.makedirs(output_folder_path, exist_ok=True)
        print(f"Image: {filename} -> Copy to folder: {output_folder_path}")
        destination_file_path = os.path.join(output_folder_path, filename)
        if not os.path.exists(destination_file_path):
            shutil.copy(file_path, destination_file_path)
            print(f"Copied image to {output_folder_path}/{filename}")
        else:
            print(f"Skipped image '{filename}' as it already exists in the destination folder.")
    else:
        print(f"Skipping non-image file: {filename}")

print("Image analysis and sorting complete.")
