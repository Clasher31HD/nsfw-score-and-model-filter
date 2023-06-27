import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import opennsfw2 as n2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from PyQt5.QtWidgets import QApplication, QFileDialog

# Create the application
app = QApplication([])

# Initialize variables
nsfw_folder_name = None
score_folder_name = None
model_folder_name = None

# Constants for NSFW ranges
NSFW_RANGES = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 0.9),
    (0.9, 0.95),
    (0.95, 0.99),
    (0.99, 1.0)
]

# Constants for score ranges
SCORE_RANGES = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0)
]


# Function to check the range for a value and return the corresponding folder name
def get_folder_name(value, ranges):
    if value is None:
        return 'None'
    for i, (start, end) in enumerate(ranges):
        if start <= value < end:
            return f"{start}-{end}"
    return 'None'


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


def extract_model_name(file_path):
    # Extract the model name from the metadata dictionary
    image = Image.open(file_path)
    metadata = image.info
    params = metadata.get("parameters", "")
    model_start = params.find("Model:") + len("Model:")
    model_end = params.find(",", model_start)
    model_name = params[model_start:model_end].strip()
    return model_name


while True:
    mode = input("Enter Mode:\n1 = NSFW\n2 = Score\n3 = Model\n4 = NSFW/Model\n5 = NSFW/Score\n6 = Score/NSFW\n7 = "
                 "Score/Model\n8 = Model/NSFW\n9 = Model/Score\n10 = NSFW/Score/Model\n11 = NSFW/Model/Score\n12 = "
                 "Score/NSFW/Model\n13 = Score/Model/NSFW\n14 = Model/NSFW/Score\n15 = Model/Score/NSFW\nSelected "
                 "Mode: ")

    if mode in [str(i) for i in range(1, 16)]:
        break

    print("Invalid mode selected. Please try again.\n")

# Define input directory
while True:
    print("Choose your input folder")
    input_folder = QFileDialog.getExistingDirectory(None, "Select input folder")

    if input_folder:
        break

# Convert input_folder to a Path object
input_folder = Path(input_folder)

# Define output directory
while True:
    print("Choose your output folder")
    output_folder = QFileDialog.getExistingDirectory(None, "Select output folder")

    if output_folder:
        break

# Convert output_folder to a Path object
output_folder = Path(output_folder)

while True:
    move_or_copy = input("Do you wanna move or copy the files?\n1 = Move\n2 = Copy\nSelected mode: ")

    if move_or_copy == "1" or move_or_copy == "2":
        break

    print("Invalid mode selected. Please try again.\n")

# Create output folder if it doesn't exist
output_folder.mkdir(parents=True, exist_ok=True)

# Count the total number of images in the input folder
valid_extensions = ('.png', '.jpg', '.jpeg')
image_files = [file_path for file_path in input_folder.rglob('*') if file_path.suffix.lower() in valid_extensions]
total_images = len(image_files)

if mode in ["2", "4", "6", "7", "9", "10", "11", "12", "13", "14", "15"]:
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

for idx, file_path in enumerate(image_files):
    # Check if the file is a valid image
    if file_path.suffix.lower() in valid_extensions:
        print(f"\nAnalyzing image {idx+1}/{total_images}\n{file_path.name}")
        if mode in ["1", "4", "5", "6", "8", "10", "11", "12", "13", "14", "15"]:
            # Check if the image is NSFW
            nsfw_probability = is_nsfw(file_path)
            print(f"NSFW probability: {nsfw_probability}")
            nsfw_folder_name = get_folder_name(nsfw_probability, NSFW_RANGES)
            nsfw_folder_name = "X" + nsfw_folder_name

        if mode in ["2", "4", "6", "7", "9", "10", "11", "12", "13", "14", "15"]:
            # Get the score and index for the input image
            class_index, score = get_image_score(file_path)
            print(f"Score: {score}, Class: {class_index}")
            score_folder_name = get_folder_name(score, SCORE_RANGES)
            score_folder_name = "S" + score_folder_name

        if mode in ["3", "5", "7", "8", "9", "10", "11", "12", "13", "14", "15"]:
            # Extract the model name
            model_name = extract_model_name(file_path)
            print(f"Model: {model_name}")
            model_folder_name = model_name

        mode_folders = {
            "1": [nsfw_folder_name],
            "2": [score_folder_name],
            "3": [model_folder_name],
            "4": [nsfw_folder_name, score_folder_name],
            "5": [nsfw_folder_name, model_folder_name],
            "6": [score_folder_name, nsfw_folder_name],
            "7": [score_folder_name, model_folder_name],
            "8": [model_folder_name, nsfw_folder_name],
            "9": [model_folder_name, score_folder_name],
            "10": [nsfw_folder_name, score_folder_name, model_folder_name],
            "11": [nsfw_folder_name, model_folder_name, score_folder_name],
            "12": [score_folder_name, nsfw_folder_name, model_folder_name],
            "13": [score_folder_name, model_folder_name, nsfw_folder_name],
            "14": [model_folder_name, nsfw_folder_name, score_folder_name],
            "15": [model_folder_name, score_folder_name, nsfw_folder_name],
        }

        if mode in mode_folders:
            new_output_folder = output_folder
            for folder_name in mode_folders[mode]:
                new_output_folder = new_output_folder / folder_name
                new_output_folder.mkdir(parents=True, exist_ok=True)
            destination_file_path = new_output_folder / file_path.name
            if not destination_file_path.exists():
                if move_or_copy == "1":
                    print(f"Image: {file_path.name} -> Move to folder: {new_output_folder}")
                    shutil.move(file_path, destination_file_path)
                    print(f"Moved image to {destination_file_path}")
                else:
                    print(f"Image: {file_path.name} -> Copy to folder: {new_output_folder}")
                    shutil.copy(file_path, destination_file_path)
                    print(f"Copied image to {destination_file_path}")
                # break  # Break out of the loop if the image was copied

            else:
                print(f"Skipped image '{file_path.name}' as it already exists in the destination folder.")
        else:
            print("Invalid mode entered.")
    else:
        print(f"Skipping non-image file: {file_path.name}")

print("Image analysis and sorting complete.")
