import os
import numpy as np
from tensorflow.keras.applications import (Xception, VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2,
                                           ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet,
                                           MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetMobile,
                                           NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
                                           EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
                                           EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
                                           EfficientNetV2S, EfficientNetV2M, EfficientNetV2L, ConvNeXtTiny,
                                           ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import shutil
from PyQt5.QtWidgets import QApplication, QFileDialog
from pathlib import Path

input_folder = None
output_folder = None
model_file = None
model_type = None

print("Initializing...")

# Create the application
app = QApplication([])

# Model selection dictionary
MODEL_SELECTION = {
    1: Xception,
    2: VGG16,
    3: VGG19,
    4: ResNet50,
    5: ResNet50V2,
    6: ResNet101,
    7: ResNet101V2,
    8: ResNet152,
    9: ResNet152V2,
    10: InceptionV3,
    11: InceptionResNetV2,
    12: MobileNet,
    13: MobileNetV2,
    14: DenseNet121,
    15: DenseNet169,
    16: DenseNet201,
    17: NASNetMobile,
    18: NASNetLarge,
    19: EfficientNetB0,
    20: EfficientNetB1,
    21: EfficientNetB2,
    22: EfficientNetB3,
    23: EfficientNetB4,
    24: EfficientNetB5,
    25: EfficientNetB6,
    26: EfficientNetB7,
    27: EfficientNetV2B0,
    28: EfficientNetV2B1,
    29: EfficientNetV2B2,
    30: EfficientNetV2B3,
    31: EfficientNetV2S,
    32: EfficientNetV2M,
    33: EfficientNetV2L,
    34: ConvNeXtTiny,
    35: ConvNeXtSmall,
    36: ConvNeXtBase,
    37: ConvNeXtLarge,
    38: ConvNeXtXLarge
}


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
    img = image.load_img(image_path, input_shape)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    gender_prediction = model.predict(img)

    # Map class indices to labels
    class_labels = ['male', 'female', 'both', 'neither']
    predicted_label = class_labels[np.argmax(gender_prediction)]

    return predicted_label


def invalid_input():
    print("Invalid input. Please try again.\n")


if model_type is None:
    while True:
        model_type = input("Which Scoring Model do you wanna use?\n1 = Xception\n2 = VGG16\n3 = VGG19\n4 = "
                           "ResNet50\n5 = ResNet50V2\n6 = ResNet101\n7 = ResNet101V2\n8 = ResNet152\n9 = "
                           "ResNet152V2\n10 = InceptionV3\n11 = InceptionResNetV2\n12 = MobileNet\13 = "
                           "MobileNetV2\n14 = DenseNet121\n15 = DenseNet169\n16 = DenseNet201\n17 = "
                           "NASNetMobile\n18 = NASNetLarge\n19 = EfficientNetB0\n20 = EfficientNetB1\n21 = "
                           "EfficientNetB2\n22 = EfficientNetB3\n23 = EfficientNetB4\n24 = EfficientNetB5\n25 = "
                           "EfficientNetB6\n26 = EfficientNetB7\n27 = EfficientNetV2B0\n28 = EfficientNetV2B1\n29 "
                           "= EfficientNetV2B2\n30 = EfficientNetV2B3\n31 = EfficientNetV2S\n32 = "
                           "EfficientNetV2M\n33 = EfficientNetV2L\n34 = ConvNeXtTiny\n35 = ConvNeXtSmall\n36 = "
                           "ConvNeXtBase\n37 = ConvNeXtLarge\n38 = ConvNeXtXLarge\nSelected Scoring Model: ")

        if model_type in [str(i) for i in range(1, 39)]:
            model_type = int(model_type)
            break
        invalid_input()

# Determine the input shape based on the selected model_type
if model_type in [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 19, 27, 34, 35, 36, 37, 38]:
    input_shape = (224, 224, 3)
elif model_type in [20, 28]:
    input_shape = (240, 240, 3)
elif model_type in [21, 29]:
    input_shape = (260, 260, 3)
elif model_type in [1, 10, 11]:
    input_shape = (299, 299, 3)
elif model_type in [22, 30]:
    input_shape = (300, 300, 3)
elif model_type in [18]:
    input_shape = (331, 331, 3)
elif model_type in [23]:
    input_shape = (380, 380, 3)
elif model_type in [31]:
    input_shape = (384, 384, 3)
elif model_type in [24]:
    input_shape = (456, 456, 3)
elif model_type in [32, 33]:
    input_shape = (480, 480, 3)
elif model_type in [25]:
    input_shape = (528, 528, 3)
elif model_type in [26]:
    input_shape = (600, 600, 3)
else:
    exit("Error 1")

print("Loading model: " + MODEL_SELECTION[int(model_type)])
# Load the pre-trained model with the determined input shape
base_model = MODEL_SELECTION[int(model_type)](weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers for gender classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # Adjust the output layer for 4 classes

model = Model(inputs=base_model.input, outputs=predictions)

print("Model loaded")

# Add code to select the model file
if model_file is None:
    model_file = get_file_path("Choose your model file")

print("Loading model: " + MODEL_SELECTION[int(model_type)])
# Load the trained weights for gender classification
model.load_weights(str(model_file))
print("Model loaded")

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
        if gender == 'male':
            destination = output_male_dir
        elif gender == 'female':
            destination = output_female_dir
        elif gender == 'both':
            destination = output_both_dir
        else:
            destination = output_neither_dir

        # Copy the image to the appropriate destination folder
        shutil.copy(image_path, os.path.join(destination, filename))

print("Gender classification complete.")
exit()
