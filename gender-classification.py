import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import shutil
from PyQt5.QtWidgets import QApplication, QFileDialog
from pathlib import Path

input_folder = None
output_folder = None
model_file = None

# Create the application
app = QApplication([])


def get_folder_path(message):
    while True:
        print(message)
        folder_path_input = QFileDialog().getExistingDirectory(None, message)

        if folder_path_input:
            folder_path = Path(folder_path_input)
            return folder_path
        else:
            print("Invalid input selected. Please try again.\n")


def get_file_path(message):
    while True:
        print(message)
        file_path_input, _ = QFileDialog.getOpenFileName(None, message, filter="Model files (*.h5);;All files (*)")

        if file_path_input:
            file_path = Path(file_path_input)
            return file_path
        else:
            print("Invalid input selected. Please try again.\n")


# Define a function to predict gender
def predict_gender(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    gender_prediction = model.predict(img)

    # Map class indices to labels
    class_labels = ['Male', 'Female', 'Both', 'Neither']
    predicted_label = class_labels[np.argmax(gender_prediction)]

    return predicted_label


# Load the pre-trained MobileNetV2 model (excluding the top layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 4))

# Add custom layers for gender classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Adjust the output layer for 3 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Add code to select the model file
if model_file is None:
    model_file = get_file_path("Choose the model file")

# Load the trained weights for gender classification
model.load_weights(str(model_file))

# Define input directory
if input_folder is None:
    input_folder = get_folder_path("Choose your input folder")

# Define input directory
if output_folder is None:
    output_folder = get_folder_path("Choose your output folder")

# Create output directories if they don't exist
output_male_dir = os.path.join(output_folder, 'male')
output_female_dir = os.path.join(output_folder, 'female')
output_both_dir = os.path.join(output_folder, 'both')
output_neither_dir = os.path.join(output_folder, 'neither')

os.makedirs(output_male_dir, exist_ok=True)
os.makedirs(output_female_dir, exist_ok=True)
os.makedirs(output_both_dir, exist_ok=True)
os.makedirs(output_neither_dir, exist_ok=True)

# Iterate through the images and categorize and copy them
for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
        image_path = os.path.join(input_folder, filename)
        gender = predict_gender(image_path)

        # Determine the destination folder
        if gender == 'Male':
            destination = output_male_dir
        elif gender == 'Female':
            destination = output_female_dir
        elif gender == 'Both':
            destination = output_both_dir
        else:
            destination = output_neither_dir

        # Copy the image to the appropriate destination folder
        shutil.copy(image_path, os.path.join(destination, filename))
