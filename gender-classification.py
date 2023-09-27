import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import shutil

# Load the pre-trained MobileNetV2 model (excluding the top layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for gender classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Adjust the output layer for 3 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Load the trained weights for gender classification
model.load_weights('gender_classification_model.h5')

# Define a function to predict gender
def predict_gender(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    gender_prediction = model.predict(img)

    # Map class indices to labels
    class_labels = ['Male', 'Female', 'Neither']
    predicted_label = class_labels[np.argmax(gender_prediction)]

    return predicted_label

# Path to the folder containing full-body images
image_folder = 'C:/Users/I539356/Downloads/www.freepik.com/People Vectors- Photos and PSD files - Free Download - 27-09-2023 11-28-54'

# Output directory where classified images will be copied
output_directory = 'C:/Users/I539356/Downloads/Gendered'

# Create output directories if they don't exist
output_male_dir = os.path.join(output_directory, 'male')
output_female_dir = os.path.join(output_directory, 'female')
output_neither_dir = os.path.join(output_directory, 'neither')

os.makedirs(output_male_dir, exist_ok=True)
os.makedirs(output_female_dir, exist_ok=True)
os.makedirs(output_neither_dir, exist_ok=True)

# Iterate through the images and categorize and copy them
for filename in os.listdir(image_folder):
    if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
        image_path = os.path.join(image_folder, filename)
        gender = predict_gender(image_path)

        # Determine the destination folder
        if gender == 'Male':
            destination = output_male_dir
        elif gender == 'Female':
            destination = output_female_dir
        else:
            destination = output_neither_dir

        # Copy the image to the appropriate destination folder
        shutil.copy(image_path, os.path.join(destination, filename))
